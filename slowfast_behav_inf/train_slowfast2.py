import os
import argparse
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, IterableDataset
from torchvision.transforms import v2 as transforms
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.cuda.amp import GradScaler, autocast
import shutil
import cv2

from typing import List, Dict, Iterator
import random

from pytorchvideo.data import LabeledVideoDataset, make_clip_sampler

from pytorchvideo.transforms import (
    ApplyTransformToKey,
    Normalize,
)

from tqdm import tqdm

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, f1_score


class StochasticStratifiedVideoDatasetWrapper(IterableDataset):
    def __init__(self, video_dataset: LabeledVideoDataset, sampling_percent: float):
        super().__init__()
        self.video_dataset = video_dataset
        self.sampling_percent = sampling_percent

        original_labeled_videos = self.video_dataset._labeled_videos
        self.class_groups: Dict[int, List] = {}
        for video_path, info_dict in original_labeled_videos:
            label = info_dict["label"]
            if label not in self.class_groups:
                self.class_groups[label] = []
            self.class_groups[label].append((video_path, info_dict))

    def __iter__(self) -> Iterator:
        print("\n   -> (Wrapper) Generating new stochastic subset for this epoch...")
        
        subset_video_paths = []
        for label, video_list in self.class_groups.items():
            num_to_sample = max(1, round(len(video_list) * self.sampling_percent))
            subset_video_paths.extend(random.sample(video_list, num_to_sample))
        
        random.shuffle(subset_video_paths)
        
        epoch_dataset = LabeledVideoDataset(
            labeled_video_paths=subset_video_paths,
            clip_sampler=self.video_dataset._clip_sampler,
            transform=self.video_dataset._transform,
            decode_audio=False,
        )
        
        return iter(epoch_dataset)


class PackPathway(torch.nn.Module):
    def __init__(self, alpha: int):
        super().__init__()
        self.alpha = alpha

    def forward(self, frames: torch.Tensor) -> List[torch.Tensor]:
        fast_pathway = frames
        slow_pathway = torch.index_select(
            frames,
            1,
            torch.linspace(
                0, frames.shape[1] - 1, frames.shape[1] // self.alpha
            ).long().to(frames.device),
        )
        frame_list = [slow_pathway, fast_pathway]
        return frame_list


class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.reduction = reduction
        if alpha is not None:
            if isinstance(alpha, (list, np.ndarray)):
                self.alpha = torch.tensor(alpha, dtype=torch.float)
            else:
                self.alpha = alpha
        else:
            self.alpha = None

    def forward(self, inputs, targets):
        CE_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-CE_loss)
        F_loss = (1 - pt)**self.gamma * CE_loss

        if self.alpha is not None:
            if self.alpha.device != inputs.device:
                self.alpha = self.alpha.to(inputs.device)
            alpha_t = self.alpha.gather(0, targets.view(-1))
            F_loss = alpha_t * F_loss

        if self.reduction == 'mean':
            return F_loss.mean()
        elif self.reduction == 'sum':
            return F_loss.sum()
        else:
            return F_loss


class PadOrClipVideo(torch.nn.Module):
    def __init__(self, target_frames: int):
        super().__init__()
        self.target_frames = target_frames

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        current_frames = frames.shape[1]
        if current_frames > self.target_frames:
            return frames[:, :self.target_frames, :, :]
        elif current_frames < self.target_frames:
            last_frame = frames[:, -1:, :, :]
            padding_needed = self.target_frames - current_frames
            padded_frames = last_frame.repeat(1, padding_needed, 1, 1)
            return torch.cat([frames, padded_frames], dim=1)
        else:
            return frames


def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def _make_labeled_video_paths(data_path, split):
    labeled_video_paths = []
    split_path = os.path.join(data_path, split)
    if not os.path.isdir(split_path):
        raise ValueError(f"Directory not found: {split_path}")

    class_names = sorted([d for d in os.listdir(split_path) if os.path.isdir(os.path.join(split_path, d))])
    class_to_idx = {class_name: i for i, class_name in enumerate(class_names)}
    class_counts = [0] * len(class_names)
    
    for class_name, class_idx in class_to_idx.items():
        class_path = os.path.join(split_path, class_name)
        for fname in os.listdir(class_path):
            if fname.lower().endswith(".mp4"):
                video_path = os.path.join(class_path, fname)
                info_dict = {"label": class_idx}
                labeled_video_paths.append((video_path, info_dict))
                class_counts[class_idx] += 1
    
    return labeled_video_paths, class_names, class_counts


def create_slowfast_model(model_arch, num_classes, unfreeze_blocks=[]):
    print(f"### INFO: Loading pre-trained {model_arch} model... ###")
    model = torch.hub.load("facebookresearch/pytorchvideo", model_arch, pretrained=True, trust_repo=True)

    for param in model.parameters():
        param.requires_grad = False

    if unfreeze_blocks:
        print(f"### INFO: Unfreezing layers in blocks {unfreeze_blocks} for fine-tuning... ###")
        for block_num in unfreeze_blocks:
            block_name = f"blocks.{block_num}"
            for name, param in model.named_parameters():
                if name.startswith(block_name):
                    param.requires_grad = True

    try:
        original_head = model.blocks[6].proj
        in_features = original_head.in_features
    except (AttributeError, IndexError):
        raise RuntimeError("Could not determine input features for the classification head. Please inspect your model architecture.")

    new_head = nn.Sequential(
        nn.Linear(in_features, 512),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.5),
        nn.Linear(512, 256),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.5),
        nn.Linear(256, num_classes)
    )

    for module in new_head:
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.01)
            torch.nn.init.constant_(module.bias, 0)

    model.blocks[6].proj = new_head
    
    print(f"### INFO: Model head replaced with a deeper head (3 FC layers, p=0.5 dropout). New number of classes: {num_classes} ###")
    return model


def create_datasets_and_val_loader(data_path, config, args):
    clip_duration_frames = config['DATA']['CLIP_DURATION_FRAMES']
    video_fps = args.fps
    clip_duration_seconds = clip_duration_frames / video_fps
    
    alpha = config.get('MODEL', {}).get('ALPHA', 4)
    batch_size = args.new_batch_size
    
    train_transform = ApplyTransformToKey(
        key="video",
        transform=transforms.Compose(
            [
                PadOrClipVideo(clip_duration_frames),
                transforms.Lambda(lambda x: x / 255.0),
                transforms.Lambda(lambda x: x.permute(1, 0, 2, 3)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=10),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                transforms.RandomApply([transforms.Grayscale(num_output_channels=3)], p=0.1),
                transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 0.1)),
                transforms.Lambda(lambda x: x + torch.randn_like(x) * 0.01),
                transforms.Lambda(lambda x: x * (1 + torch.randn_like(x) * 0.05)),
                transforms.RandomErasing(p=0.25, scale=(0.02, 0.2), ratio=(0.3, 3.3)),
                transforms.Lambda(lambda x: x.permute(1, 0, 2, 3)),
                transforms.Resize((224, 224)),
                Normalize((0.45, 0.45, 0.45), (0.225, 0.225, 0.225)),
                PackPathway(alpha=alpha)
            ]
        ),
    )
    
    val_transform = ApplyTransformToKey(
        key="video",
        transform=transforms.Compose(
            [
                PadOrClipVideo(clip_duration_frames),
                transforms.Lambda(lambda x: x / 255.0),
                transforms.Resize((224, 224)),
                Normalize((0.45, 0.45, 0.45), (0.225, 0.225, 0.225)),
                PackPathway(alpha=alpha)
            ]
        ),
    )

    train_video_paths, class_names, train_class_counts = _make_labeled_video_paths(data_path, "train")
    val_video_paths, _, _ = _make_labeled_video_paths(data_path, "test")

    train_dataset = LabeledVideoDataset(
        labeled_video_paths=train_video_paths,
        clip_sampler=make_clip_sampler("random", clip_duration_seconds),
        transform=train_transform,
        decode_audio=False,
    )

    val_dataset = LabeledVideoDataset(
        labeled_video_paths=val_video_paths,
        clip_sampler=make_clip_sampler("uniform", clip_duration_seconds),
        transform=val_transform,
        decode_audio=False,
    )

    num_workers = args.num_workers if args.num_workers is not None else config['DATA']['NUM_WORKERS']
    print(f"### INFO: Using {num_workers} workers for data loading. ###")

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )
    
    return train_dataset, val_loader, class_names, train_class_counts


def train_one_epoch(model, dataloader, optimizer, criterion, device, scaler, use_amp):
    model.train()
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    progress_bar = tqdm(dataloader, desc="Training", unit="batch")

    for batch in progress_bar:
        videos = batch["video"]
        labels = batch["label"]

        videos = [v.to(device) for v in videos]
        labels = labels.to(device)

        optimizer.zero_grad()

        with autocast(enabled=use_amp):
            preds = model(videos)
            loss = criterion(preds, labels)

        scaler.scale(loss).backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        scaler.step(optimizer)
        
        scaler.update()

        running_loss += loss.item() * videos[0].size(0)
        _, predicted_labels = torch.max(preds, 1)
        correct_predictions += (predicted_labels == labels).sum().item()
        total_samples += labels.size(0)
        
        current_acc = correct_predictions / total_samples if total_samples > 0 else 0
        progress_bar.set_postfix(loss=loss.item(), acc=f"{current_acc:.4f}", lr=optimizer.param_groups[0]['lr'])

    epoch_loss = running_loss / total_samples if total_samples > 0 else 0
    epoch_acc = correct_predictions / total_samples if total_samples > 0 else 0
    return epoch_loss, epoch_acc


def evaluate(model, dataloader, criterion, device, use_amp):
    model.eval()
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    all_labels = []
    all_predictions = []
    all_pred_probs = []

    progress_bar = tqdm(dataloader, desc="Evaluating", unit="batch")
    
    with torch.no_grad():
        for batch in progress_bar:
            videos = batch["video"]
            labels = batch["label"]

            videos = [v.to(device) for v in videos]
            labels = labels.to(device)

            with autocast(enabled=use_amp):
                preds = model(videos)
                loss = criterion(preds, labels)
            
            probabilities = torch.softmax(preds, dim=1)
            pred_probs, predicted_labels = torch.max(probabilities, 1)

            running_loss += loss.item() * videos[0].size(0)
            correct_predictions += (predicted_labels == labels).sum().item()
            total_samples += labels.size(0)
            
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted_labels.cpu().numpy())
            all_pred_probs.extend(pred_probs.cpu().numpy())
            
            current_acc = correct_predictions / total_samples if total_samples > 0 else 0
            progress_bar.set_postfix(loss=loss.item(), acc=f"{current_acc:.4f}")

    epoch_loss = running_loss / total_samples if total_samples > 0 else 0
    epoch_acc = correct_predictions / total_samples if total_samples > 0 else 0
    epoch_f1 = f1_score(all_labels, all_predictions, average='weighted')

    return epoch_loss, epoch_acc, epoch_f1, all_labels, all_predictions, all_pred_probs


def visualize_misclassifications(output_dir, val_dataset, class_names, true_labels, pred_labels, pred_probs):
    print("### INFO: Visualizing misclassifications for the best model... ###")
    vis_path = os.path.join(output_dir, "off_diagonal_visualization")

    if os.path.exists(vis_path):
        shutil.rmtree(vis_path)
    os.makedirs(vis_path)

    video_paths = [item[0] for item in val_dataset._labeled_videos]
    misclassified_count = 0
    for i in range(len(true_labels)):
        if true_labels[i] != pred_labels[i]:
            misclassified_count += 1
            true_class = class_names[true_labels[i]]
            pred_class = class_names[pred_labels[i]]
            
            prob = pred_probs[i]
            prob_str = f"{int(prob * 10000):04d}"
            
            target_folder_name = f"{true_class}_pred_{pred_class}"
            target_dir = os.path.join(vis_path, target_folder_name)
            os.makedirs(target_dir, exist_ok=True)
            
            source_video_path = video_paths[i]
            original_filename = os.path.basename(source_video_path)
            name, ext = os.path.splitext(original_filename)
            new_filename = f"{name}_prob_{prob_str}_true_{true_class}_pred_{pred_class}{ext}"
            
            dest_video_path = os.path.join(target_dir, new_filename)
            shutil.copy(source_video_path, dest_video_path)
    
    print(f"### INFO: Saved {misclassified_count} misclassified video clips to '{vis_path}' ###")


def run_training_stage(stage_name, current_unfreeze_blocks, base_output_dir, initial_checkpoint, args, config, train_loader, val_loader, class_names, train_class_counts, device):
    stage_output_dir = os.path.join(base_output_dir, stage_name)
    os.makedirs(stage_output_dir, exist_ok=True)
    print(f"\n--- Starting Training Stage: {stage_name} ---")
    print(f"Output will be saved to: {stage_output_dir}")
    print(f"Unfrozen blocks for this stage: {current_unfreeze_blocks}")

    num_classes = len(class_names)
    
    class_weights = None
    if sum(train_class_counts) > 0:
        total_samples_full_dataset = sum(train_class_counts)
        class_weights_np = total_samples_full_dataset / (np.array(train_class_counts) * num_classes)
        class_weights = torch.tensor(class_weights_np, dtype=torch.float32).to(device)
        class_weights = class_weights / class_weights.sum() * num_classes
    else:
        print("!!! WARNING: No training samples found. Cannot calculate class weights. !!!")

    focal_loss_alpha_param = None
    if args.focal_loss_alpha:
        try:
            alpha_str = args.focal_loss_alpha.strip('[]')
            focal_loss_alpha_param = [float(x) for x in alpha_str.split()]
            if len(focal_loss_alpha_param) != num_classes:
                raise ValueError(f"Alpha values count must match number of classes.")
            focal_loss_alpha_param = torch.tensor(focal_loss_alpha_param, dtype=torch.float32).to(device)
        except Exception as e:
            print(f"!!! ERROR: Error parsing --focal-loss-alpha: {e}. Falling back. !!!")
            focal_loss_alpha_param = None
    
    if focal_loss_alpha_param is None and class_weights is not None:
        focal_loss_alpha_param = class_weights

    if args.use_focal_loss:
        if args.label_smooth > 0.0:
             print(f"!!! WARNING: --label-smooth is set to {args.label_smooth} but is only applicable to CrossEntropyLoss. It will be ignored because --use-focal-loss is active. !!!")
        criterion = FocalLoss(alpha=focal_loss_alpha_param, gamma=args.focal_loss_gamma)
        print(f"Using Focal Loss with gamma={args.focal_loss_gamma} and alpha={focal_loss_alpha_param}")
    else:
        criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=args.label_smooth)
        if args.label_smooth > 0:
            print(f"Using Cross-Entropy Loss with class weights and Label Smoothing ({args.label_smooth})")
        else:
            print(f"Using Cross-Entropy Loss with class weights: {class_weights}")
    
    model = create_slowfast_model(config['MODEL']['ARCH'], num_classes, unfreeze_blocks=current_unfreeze_blocks)
    model.to(device)

    if initial_checkpoint and os.path.exists(initial_checkpoint):
        print(f"### INFO: Loading model from checkpoint: {initial_checkpoint} ###")
        checkpoint = torch.load(initial_checkpoint, map_location=device)
        model.load_state_dict(checkpoint.get('model_state_dict', checkpoint))
        print("@@@ SUCCESS: Model state loaded successfully. @@@")
    elif initial_checkpoint:
        print(f"!!! WARNING: Initial checkpoint not found. Starting from scratch for this stage. !!!")

    head_params = [p for n, p in model.named_parameters() if "blocks.6.proj" in n and p.requires_grad]
    backbone_params = [p for n, p in model.named_parameters() if "blocks.6.proj" not in n and p.requires_grad]

    head_lr = args.new_lr
    backbone_lr = args.lr_backbone if args.lr_backbone is not None else head_lr * 0.1

    print(f"### INFO: Using AdamW optimizer with differential learning rates: ###")
    print(f"  - Classification Head LR: {head_lr}")
    if backbone_params:
        print(f"  - Unfrozen Backbone LR: {backbone_lr}")
    else:
        print("  - No backbone layers are unfrozen.")

    optimizer_params = [{'params': head_params, 'lr': head_lr}]
    if backbone_params:
        optimizer_params.append({'params': backbone_params, 'lr': backbone_lr})
    
    optimizer = optim.AdamW(optimizer_params, weight_decay=5e-4)

    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='max',
        factor=0.2,
        patience=args.lr_patience,
        threshold=args.min_delta,
        verbose=True
    )

    use_amp = args.use_amp and device.type == 'cuda'
    scaler = GradScaler(enabled=use_amp)
    if use_amp:
        print("### INFO: Automatic Mixed Precision (AMP) is ENABLED. ###")

    best_val_f1 = -float('inf')
    epochs_no_improve = 0
    best_model_path_current_stage = None

    print("\n--- Starting Training Loop for Current Stage ---")
    for epoch in range(config['TRAIN']['EPOCHS']):
        print(f"\nEpoch {epoch+1}/{config['TRAIN']['EPOCHS']} (Stage: {stage_name})")
        
        if epoch < args.warmup_epochs:
            start_lr = 1e-8
            
            target_lr_head = head_lr
            current_lr_head = start_lr + (target_lr_head - start_lr) * (epoch + 1) / args.warmup_epochs
            optimizer.param_groups[0]['lr'] = current_lr_head
            
            if len(optimizer.param_groups) > 1:
                target_lr_backbone = backbone_lr
                current_lr_backbone = start_lr + (target_lr_backbone - start_lr) * (epoch + 1) / args.warmup_epochs
                optimizer.param_groups[1]['lr'] = current_lr_backbone
                backbone_lr_str = f"{optimizer.param_groups[1].get('lr', 0):.2e}"
            else:
                backbone_lr_str = "N/A"

            print(f"### INFO: Warm-up Epoch {epoch+1}/{args.warmup_epochs}: Head LR set to {current_lr_head:.2e}, Backbone LR set to {backbone_lr_str} ###")

        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device, scaler, use_amp)
        print(f"Epoch {epoch+1} Training -> Loss: {train_loss:.4f}, Accuracy: {train_acc:.4f}")

        val_loss, val_acc, val_f1, all_labels, all_predictions, all_pred_probs = evaluate(model, val_loader, criterion, device, use_amp)
        
        cm = confusion_matrix(all_labels, all_predictions)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
        plt.title(f"Confusion Matrix - Epoch {epoch+1}")
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.tight_layout()
        cm_filename = os.path.join(stage_output_dir, f"confusion_matrix_epoch_{epoch+1}.png")
        plt.savefig(cm_filename)
        plt.close()
        print(f"Confusion matrix saved to {cm_filename}")
        
        print(f"Epoch {epoch+1} Validation -> Loss: {val_loss:.4f}, Accuracy: {val_acc:.4f}, F1-score: {val_f1:.4f}")

        if epoch >= args.warmup_epochs:
            scheduler.step(val_f1)

        if val_f1 > best_val_f1 + args.min_delta:
            best_val_f1 = val_f1
            epochs_no_improve = 0
            
            f1_int = int(best_val_f1 * 10000)
            save_filename = f"best_model_{f1_int:04d}.pth"
            save_path = os.path.join(stage_output_dir, save_filename)
            
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'best_val_f1': best_val_f1,
                'unfrozen_blocks': current_unfreeze_blocks
            }, save_path)
            best_model_path_current_stage = save_path
            print("\n" + "@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
            print(f"@@@ SUCCESS: Best model for stage '{stage_name}' saved with F1: {best_val_f1:.4f} to {save_filename} @@@")
            print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@" + "\n")

            if args.confusion_off_diag_vis:
                visualize_misclassifications(stage_output_dir, val_loader.dataset, class_names, all_labels, all_predictions, all_pred_probs)
        else:
            epochs_no_improve += 1
            print(f"No significant improvement for {epochs_no_improve} epochs.")
            if epochs_no_improve >= args.patience:
                print(f"!!! WARNING: Early stopping triggered for stage '{stage_name}'. !!!")
                break
    
    print(f"--- Training Stage '{stage_name}' Finished ---")
    print(f"Best validation F1-score for this stage: {best_val_f1:.4f}")
    return best_model_path_current_stage


def main():
    cv2.setNumThreads(0)
    
    parser = argparse.ArgumentParser(description="Train a SlowFast model using PyTorchVideo.")
    parser.add_argument("--config", type=str, required=True, help="Path to the YAML configuration file.")
    parser.add_argument("--data-path", type=str, required=True, help="Path to the root data directory.")
    parser.add_argument("--output-dir", type=str, default="training_output_pytorchvideo", help="Base directory to save outputs.")
    parser.add_argument("--new-lr", type=float, default=1e-5, help="New learning rate for the classification head.")
    parser.add_argument("--new-batch-size", type=int, default=64, help="New batch size for training.")
    parser.add_argument("--fps", type=int, default=30, help="Manually override the video FPS.")
    parser.add_argument("--unfreeze-blocks", type=int, nargs='+', default=[], help="List of block numbers to progressively unfreeze.")
    parser.add_argument("--lr-backbone", type=float, default=1e-6, help="Learning rate for the unfrozen backbone layers.")
    parser.add_argument("--use-focal-loss", action="store_true", help="Use Focal Loss instead of Cross-Entropy.")
    parser.add_argument("--focal-loss-gamma", type=float, default=2.0, help="Gamma for Focal Loss.")
    parser.add_argument("--focal-loss-alpha", type=str, default=None, help="Alpha (class weights) for Focal Loss as a string e.g. '[0.1 0.9]'.")
    parser.add_argument("--confusion-off-diag-vis", action="store_true", help="Enable saving of misclassified video clips.")
    parser.add_argument("--patience", type=int, default=20, help="Epochs to wait for improvement before early stopping.")
    parser.add_argument("--min-delta", type=float, default=0.0001, help="Minimum change in F1-score to qualify as an improvement.")
    parser.add_argument("--num-workers", type=int, default=None, help="Override number of data loading workers from config.")
    parser.add_argument("--stoch-sampling-percent", type=float, default=1.0, help="Percentage of training data to stochastically sample per epoch (0.0 to 1.0). 1.0 uses all data.")
    parser.add_argument("--label-smooth", type=float, default=0.0, help="Value for label smoothing for CrossEntropyLoss (e.g., 0.1). Default: 0.0 (off).")
    parser.add_argument("--use-amp", action="store_true", help="Enable Automatic Mixed Precision (AMP) for faster training on compatible GPUs.")
    parser.add_argument("--warmup-epochs", type=int, default=5, help="Number of epochs for linear learning rate warm-up.")
    parser.add_argument("--lr-patience", type=int, default=5, help="Patience for ReduceLROnPlateau scheduler.")
    parser.add_argument("--resume-checkpoint", type=str, default=None, help="Path to a checkpoint file to resume training from.")
    parser.add_argument("--resume-from-block", type=int, default=None,
                        help="Start progressive unfreezing from this block number. Must be used with --resume-checkpoint.")
    parser.add_argument("--full-fine-tuning", action="store_true", 
                        help="Enable full fine-tuning mode. Skips progressive unfreezing and trains all specified --unfreeze-blocks at once.")
    args = parser.parse_args()

    if not (0.0 < args.stoch_sampling_percent <= 1.0):
        raise ValueError("--stoch-sampling-percent must be a float between 0 (exclusive) and 1 (inclusive).")
    if args.resume_from_block is not None and args.resume_checkpoint is None:
        raise ValueError("!!! ERROR: --resume-from-block must be used with --resume-checkpoint. !!!")

    config = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(config['MISC']['SEED'])
    
    os.makedirs(args.output_dir, exist_ok=True)

    train_dataset, val_loader, class_names, train_class_counts = create_datasets_and_val_loader(
        args.data_path, config, args
    )

    if args.stoch_sampling_percent < 1.0:
        print(f"### INFO: Wrapping training dataset for stratified stochastic sampling with {args.stoch_sampling_percent*100:.1f}% of the data per epoch. ###")
        train_dataset = StochasticStratifiedVideoDatasetWrapper(train_dataset, args.stoch_sampling_percent)
    else:
        print("### INFO: Using the full training dataset for each epoch. ###")
    
    num_workers = args.num_workers if args.num_workers is not None else config['DATA']['NUM_WORKERS']

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.new_batch_size,
        sampler=None,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    if args.full_fine_tuning:
        print(f"\n{'='*80}\n{' '*20} MODE: Full Fine-tuning {' '*25}\n{'='*80}")

        unfreeze_target_blocks = sorted(list(set(args.unfreeze_blocks + [6]))) if args.unfreeze_blocks else []
        
        if not args.unfreeze_blocks:
            stage_name = "fc_only"
        else:
            stage_name = "_".join(map(str, ["full_finetune"] + sorted(unfreeze_target_blocks)))

        run_training_stage(
            stage_name=stage_name,
            current_unfreeze_blocks=unfreeze_target_blocks,
            base_output_dir=args.output_dir,
            initial_checkpoint=args.resume_checkpoint,
            args=args, config=config, train_loader=train_loader, val_loader=val_loader,
            class_names=class_names, train_class_counts=train_class_counts, device=device
        )
    else:
        print(f"\n{'='*80}\n{' '*20} MODE: Progressive Fine-tuning {' '*20}\n{'='*80}")
        current_checkpoint_path = args.resume_checkpoint
        sorted_unfreeze_blocks = sorted(args.unfreeze_blocks, reverse=True) 

        stages_to_run = []
        if args.resume_from_block is None:
            stages_to_run.append({'name': 'fc_only', 'blocks': []})
            
            unfrozen_blocks_for_stage = []
            for block in sorted_unfreeze_blocks:
                unfrozen_blocks_for_stage.append(block)
                stage_name = "_".join(map(str, ["unfreeze"] + sorted(unfrozen_blocks_for_stage)))
                stages_to_run.append({'name': stage_name, 'blocks': sorted(unfrozen_blocks_for_stage)})
        else:
            if args.resume_from_block not in sorted_unfreeze_blocks:
                raise ValueError(f"!!! ERROR: --resume-from-block value {args.resume_from_block} not found in the reverse-sorted --unfreeze-blocks list: {sorted_unfreeze_blocks} !!!")
            
            start_index = sorted_unfreeze_blocks.index(args.resume_from_block)
            
            unfrozen_blocks_for_stage = sorted_unfreeze_blocks[:start_index]
            
            for i in range(start_index, len(sorted_unfreeze_blocks)):
                block = sorted_unfreeze_blocks[i]
                unfrozen_blocks_for_stage.append(block)
                stage_name = "_".join(map(str, ["unfreeze"] + sorted(unfrozen_blocks_for_stage)))
                stages_to_run.append({'name': stage_name, 'blocks': sorted(unfrozen_blocks_for_stage)})

        for stage_info in stages_to_run:
            stage_name = stage_info['name']
            blocks_to_unfreeze = stage_info['blocks']
            
            print(f"\n{'='*80}\n{' '*20} PHASE: {stage_name} {' '*19}\n{'='*80}")
            
            if current_checkpoint_path and not os.path.exists(current_checkpoint_path):
                print(f"!!! WARNING: Checkpoint for stage '{stage_name}' not found at {current_checkpoint_path}. Halting. !!!")
                break
                
            next_checkpoint_path = run_training_stage(
                stage_name=stage_name,
                current_unfreeze_blocks=blocks_to_unfreeze,
                base_output_dir=args.output_dir,
                initial_checkpoint=current_checkpoint_path,
                args=args, config=config, train_loader=train_loader, val_loader=val_loader,
                class_names=class_names, train_class_counts=train_class_counts, device=device
            )

            if not next_checkpoint_path:
                print(f"!!! ERROR: Stage '{stage_name}' did not produce a best model. Halting subsequent stages. !!!")
                break
            
            current_checkpoint_path = next_checkpoint_path

    print("\n$$$ All training finished! $$$")


if __name__ == "__main__":
    main()

import os
import argparse
import yaml
import torch
import torch.nn as nn
import pandas as pd
from ultralytics import YOLO
from tqdm import tqdm
from typing import List, Dict
import gc
import warnings
import cv2
import numpy as np

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

from pytorchvideo.transforms import ShortSideScale, Normalize
from torchvision.transforms import v2 as transforms
from torch.cuda.amp import autocast

import torch._dynamo
torch._dynamo.config.suppress_errors = True

class PackPathway(torch.nn.Module):
    def __init__(self, alpha: int):
        super().__init__()
        self.alpha = alpha

    def forward(self, frames: torch.Tensor) -> List[torch.Tensor]:
        fast_pathway = frames
        slow_pathway = torch.index_select(
            frames, 1,
            torch.linspace(0, frames.shape[1] - 1, frames.shape[1] // self.alpha).long().to(frames.device),
        )
        return [slow_pathway, fast_pathway]

def filter_short_events(predictions: np.ndarray, min_duration: int) -> np.ndarray:
    if min_duration <= 1:
        return predictions
    
    filtered_predictions = predictions.copy()
    n_frames = len(predictions)
    if n_frames == 0:
        return predictions

    change_indices = np.where(predictions[:-1] != predictions[1:])[0] + 1
    start_indices = np.concatenate(([0], change_indices))
    end_indices = np.concatenate((change_indices, [n_frames]))

    first_stable_label = predictions[0]
    for start, end in zip(start_indices, end_indices):
        if (end - start) >= min_duration:
            first_stable_label = predictions[start]
            break

    last_valid_label = first_stable_label
    for start, end in zip(start_indices, end_indices):
        if (end - start) < min_duration:
            filtered_predictions[start:end] = last_valid_label
        else:
            last_valid_label = predictions[start]
            
    return filtered_predictions

def load_slowfast_config(config_path: str) -> dict:
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    print("### INFO: SlowFast configuration loaded successfully. ###")
    return config

def create_slowfast_model(model_arch: str, num_classes: int) -> nn.Module:
    print(f"### INFO: Loading pre-trained SlowFast model: {model_arch} ###")
    model = torch.hub.load("facebookresearch/pytorchvideo", model_arch, pretrained=True, trust_repo=True)
    
    for param in model.parameters():
        param.requires_grad = False
    
    original_head = model.blocks[6].proj
    in_features = original_head.in_features

    new_head = nn.Sequential(
        nn.Linear(in_features, 512), nn.ReLU(inplace=True), nn.Dropout(p=0.3),
        nn.Linear(512, 256), nn.ReLU(inplace=True), nn.Dropout(p=0.3),
        nn.Linear(256, num_classes)
    )

    for module in new_head:
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.01)
            torch.nn.init.constant_(module.bias, 0)

    model.blocks[6].proj = new_head
    print(f"### INFO: SlowFast model head replaced. New number of classes: {num_classes} ###")
    return model

def get_slowfast_transforms(config: dict) -> transforms.Compose:
    alpha = config.get('MODEL', {}).get('ALPHA', 4)
    return transforms.Compose([
        transforms.Lambda(lambda x: x / 255.0),
        Normalize((0.45, 0.45, 0.45), (0.225, 0.225, 0.225)),
        ShortSideScale(size=256),
        transforms.CenterCrop(224),
        PackPathway(alpha=alpha)
    ])

def get_class_names(data_path: str) -> List[str]:
    train_path = os.path.join(data_path, "train")
    if not os.path.isdir(train_path):
        raise ValueError(f"Training data directory not found at {train_path}.")
    
    class_names = sorted([d for d in os.listdir(train_path) if os.path.isdir(os.path.join(train_path, d))])
    print(f"### INFO: Detected behavior class names: {class_names} ###")
    return class_names

def process_sf_batch(
    batch: List[Dict], model: nn.Module, transforms: transforms.Compose, device: torch.device,
    behavior_predictions: np.ndarray, all_frame_probabilities: np.ndarray, behavior_thresholds: List[float], use_amp: bool
):
    if not batch:
        return

    with torch.no_grad():
        clip_tensors_cpu = [item['tensor'] for item in batch]
        center_indices = [item['center_idx'] for item in batch]
        
        transformed_batch = [transforms(clip.to(device)) for clip in clip_tensors_cpu]
        
        slow_pathways = torch.stack([item[0] for item in transformed_batch])
        fast_pathways = torch.stack([item[1] for item in transformed_batch])

        with autocast(enabled=use_amp):
            predictions = model([slow_pathways, fast_pathways])
        
        probabilities = torch.softmax(predictions, dim=1)

        for prob_single, center_idx in zip(probabilities, center_indices):
            selected_label = -1
            if behavior_thresholds:
                sorted_probs, sorted_indices = torch.sort(prob_single, descending=True)
                for k in range(len(sorted_indices)):
                    if sorted_probs[k].item() >= behavior_thresholds[sorted_indices[k].item()]:
                        selected_label = sorted_indices[k].item()
                        break
                if selected_label == -1:
                    selected_label = torch.argmax(prob_single).item()
            else:
                selected_label = torch.argmax(prob_single).item()
            
            if 0 <= center_idx < len(behavior_predictions):
                behavior_predictions[center_idx] = selected_label
                all_frame_probabilities[center_idx] = prob_single.cpu().numpy()
    
    del clip_tensors_cpu, center_indices, transformed_batch, slow_pathways, fast_pathways, predictions, probabilities
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def run_pipeline(args: argparse.Namespace):
    device = torch.device(f'cuda:{args.gpu_id}' if torch.cuda.is_available() else 'cpu')
    print(f"### INFO: Using device: {device} ###")

    yolo_model = YOLO(args.yolo_model)
    yolo_model.to(device)
    print(f"### INFO: YOLO model '{args.yolo_model}' loaded. ###")

    slowfast_config = load_slowfast_config(args.slowfast_config)
    class_names = get_class_names(args.training_data_path)
    num_classes = len(class_names)

    slowfast_model = create_slowfast_model(slowfast_config['MODEL']['ARCH'], num_classes)
    checkpoint = torch.load(args.slowfast_model, map_location=device)
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    slowfast_model.load_state_dict(state_dict, strict=False)
    slowfast_model.eval().to(device)
    print(f"### INFO: SlowFast model '{args.slowfast_model}' loaded. ###")

    slowfast_transforms = get_slowfast_transforms(slowfast_config)
    clip_duration = slowfast_config['DATA']['CLIP_DURATION_FRAMES']
    total_frames_for_clip_span = clip_duration * args.sample_stride

    os.makedirs(args.output_folder, exist_ok=True)
    csv_dir = os.path.join(args.output_folder, "csv_labels")
    video_dir = os.path.join(args.output_folder, "video_visualizations")
    os.makedirs(csv_dir, exist_ok=True)
    os.makedirs(video_dir, exist_ok=True)

    video_files = [os.path.join(args.input_path, f) for f in os.listdir(args.input_path) if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))] if os.path.isdir(args.input_path) else [args.input_path]

    for video_path in tqdm(video_files, desc="Overall Progress", unit="video"):
        output_base_name = os.path.splitext(os.path.basename(video_path))[0]
        output_video_path = os.path.join(video_dir, f"{output_base_name}_final.mp4")
        output_csv_path = os.path.join(csv_dir, f"{output_base_name}_labels.csv")
        output_raw_csv_path = os.path.join(csv_dir, f"{output_base_name}_raw_inference.csv")

        if os.path.exists(output_video_path) and os.path.exists(output_csv_path):
            print(f"--- SKIPPING {os.path.basename(video_path)}: All output files already exist. ---")
            continue

        print(f"\n--- Processing video: {os.path.basename(video_path)} ---")
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"!!! ERROR: Could not open video file: {video_path} !!!")
            continue

        orig_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        source_fps = int(cap.get(cv2.CAP_PROP_FPS))
        output_fps = args.fps if args.fps is not None else source_fps
        frame_width = int(orig_width * args.downsample)
        frame_height = int(orig_height * args.downsample)
        print(f"### INFO: Source FPS: {source_fps}, Output FPS: {output_fps}, Output Res: {frame_width}x{frame_height} ###")

        print("--- Step 1.1: Reading all video frames into memory... ---")
        all_frames_downsampled = [cv2.resize(frame, (frame_width, frame_height), interpolation=cv2.INTER_AREA) if args.downsample != 1.0 else frame for ret, frame in iter(lambda: cap.read(), (False, None))]
        cap.release()

        print("--- Step 1.2: Running YOLO detection on all frames... ---")
        all_yolo_details = []
        for i in tqdm(range(0, len(all_frames_downsampled), args.yolo_batch_size), desc="YOLO Processing", unit="batch"):
            results = yolo_model(all_frames_downsampled[i:i + args.yolo_batch_size], verbose=False, conf=args.confidence)
            for r in results:
                best_mouse_info = None
                highest_conf = -1.0
                if r.boxes:
                    for box in r.boxes:
                        if int(box.cls[0]) == args.mouse_class_id:
                            conf = box.conf[0].item()
                            if conf > highest_conf:
                                highest_conf = conf
                                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                                best_mouse_info = {
                                    "box": (int(x1), int(y1), int(x2 - x1), int(y2 - y1)),
                                    "class_id": int(box.cls[0]), "confidence": conf,
                                    "center_xy": (int(x1 + (x2 - x1) / 2), int(y1 + (y2 - y1) / 2))
                                }
                all_yolo_details.append(best_mouse_info)

        last_known_details = None
        for i in range(len(all_yolo_details)):
            if all_yolo_details[i] is not None:
                last_known_details = all_yolo_details[i]
            elif last_known_details is not None:
                all_yolo_details[i] = last_known_details
            else:
                default_box = (frame_width//2 - args.crop_size//2, frame_height//2 - args.crop_size//2, args.crop_size, args.crop_size)
                all_yolo_details[i] = {"box": default_box, "class_id": np.nan, "confidence": np.nan, "center_xy": (frame_width//2, frame_height//2)}

        all_bounding_boxes = [d['box'] for d in all_yolo_details]
        all_bbox_centers_x = [d['center_xy'][0] for d in all_yolo_details]
        all_bbox_centers_y = [d['center_xy'][1] for d in all_yolo_details]
        all_yolo_class_ids = [d['class_id'] for d in all_yolo_details]
        all_yolo_confidences = [d['confidence'] for d in all_yolo_details]

        print("--- Step 2: Classifying behavior with SlowFast (streaming)... ---")
        behavior_predictions = np.full(len(all_frames_downsampled), np.nan)
        all_frame_probabilities = np.full((len(all_frames_downsampled), num_classes), np.nan)
        clip_batch = []

        for end_idx in tqdm(range(len(all_frames_downsampled)), desc="SlowFast Processing", unit="frame", leave=False):
            start_idx = end_idx - total_frames_for_clip_span + 1
            if start_idx >= 0 and start_idx % args.stride == 0:
                clip_frames_cropped = []
                for i in range(clip_duration):
                    frame_idx = start_idx + (i * args.sample_stride)
                    if frame_idx >= len(all_frames_downsampled): continue
                    
                    frame_to_crop = all_frames_downsampled[frame_idx]
                    box = all_bounding_boxes[frame_idx]
                    x, y, w, h = box
                    center_x, center_y = x + w // 2, y + h // 2
                    size = args.crop_size
                    start_x, start_y = center_x - size // 2, center_y - size // 2
                    
                    cropped_frame = np.zeros((size, size, 3), dtype=np.uint8)
                    fsx, fsy = max(0, start_x), max(0, start_y)
                    fex, fey = min(frame_width, start_x + size), min(frame_height, start_y + size)
                    psx, psy = max(0, -start_x), max(0, -start_y)
                    
                    if fex > fsx and fey > fsy:
                        cropped_frame[psy:psy+fey-fsy, psx:psx+fex-fsx] = frame_to_crop[fsy:fey, fsx:fex]
                    clip_frames_cropped.append(cropped_frame)

                if len(clip_frames_cropped) == clip_duration:
                    clip_tensor = torch.from_numpy(np.stack(clip_frames_cropped)).permute(3, 0, 1, 2).float()
                    center_idx = start_idx + (total_frames_for_clip_span // 2)
                    clip_batch.append({'tensor': clip_tensor, 'center_idx': center_idx})

                if len(clip_batch) >= args.slowfast_batch_size:
                    process_sf_batch(clip_batch, slowfast_model, slowfast_transforms, device, behavior_predictions, all_frame_probabilities, None, args.use_amp)
                    clip_batch.clear()
                    gc.collect()

        if clip_batch:
            process_sf_batch(clip_batch, slowfast_model, slowfast_transforms, device, behavior_predictions, all_frame_probabilities, None, args.use_amp)

        print("--- Step 3: Post-processing and saving results... ---")
        df_raw = pd.DataFrame({
            'frame_idx': np.arange(len(all_frames_downsampled)), 'raw_behavior_id': behavior_predictions,
            'yolo_bbox_center_x': all_bbox_centers_x, 'yolo_bbox_center_y': all_bbox_centers_y,
            'yolo_class_id': all_yolo_class_ids, 'yolo_confidence': all_yolo_confidences
        })
        df_raw['raw_behavior_label'] = df_raw['raw_behavior_id'].apply(lambda x: class_names[int(x)] if pd.notna(x) else "")
        df_probs = pd.DataFrame(all_frame_probabilities, columns=[f'prob_{name}' for name in class_names])
        df_raw_full = pd.concat([df_raw, df_probs], axis=1)
        df_raw_full['source_fps'] = source_fps
        df_raw_full.to_csv(output_raw_csv_path, index=False)
        print(f"@@@ SUCCESS: Saved raw inference data to {output_raw_csv_path} @@@")

        final_behaviors = filter_short_events(pd.Series(behavior_predictions).ffill().bfill().to_numpy(na_value=0).astype(int), args.min_event_duration)
        interpolated_probabilities = pd.DataFrame(all_frame_probabilities).ffill().bfill().to_numpy()

        print("--- Step 4: Generating final visualization video... ---")
        fourcc = cv2.VideoWriter_fourcc(*args.codec)
        out_video = cv2.VideoWriter(output_video_path, fourcc, output_fps, (frame_width, frame_height))
        
        behavior_colors = [(0, 255, 0), (0, 0, 255), (255, 0, 0), (0, 255, 255), (255, 255, 0), (255, 0, 255)]
        
        for idx in tqdm(range(0, len(all_frames_downsampled), args.skip_frames_for_visualization), desc="Final Visualization", unit="frame", leave=False):
            frame = all_frames_downsampled[idx].copy()
            box = all_bounding_boxes[idx]
            behavior_idx = final_behaviors[idx]
            
            label_text = class_names[behavior_idx] if 0 <= behavior_idx < len(class_names) else "Unknown"
            label_color = behavior_colors[behavior_idx % len(behavior_colors)]
            
            if box:
                prob_text = f" (P: {interpolated_probabilities[idx, behavior_idx]:.2f})"
                label_text_with_prob = f"{label_text}{prob_text}"
                font, scale, thickness = cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2
                text_size = cv2.getTextSize(label_text_with_prob, font, scale, thickness)[0]
                text_x, text_y = box[0] + box[2] - text_size[0], box[1] - 10
                if text_y < text_size[1] + 5: text_y = box[1] + text_size[1] + 5
                cv2.putText(frame, label_text_with_prob, (text_x, text_y), font, scale, label_color, thickness)
                cv2.rectangle(frame, (box[0], box[1]), (box[0] + box[2], box[1] + box[3]), label_color, 2)
            
            out_video.write(frame)
            
        out_video.release()
        
        df_labels = pd.DataFrame({
            'frame_idx': np.arange(len(all_frames_downsampled)), 'behavior_id': final_behaviors,
            'behavior_label': [class_names[i] for i in final_behaviors],
            'yolo_bbox_center_x': all_bbox_centers_x, 'yolo_bbox_center_y': all_bbox_centers_y,
            'yolo_class_id': all_yolo_class_ids, 'yolo_confidence': all_yolo_confidences
        })
        df_final = pd.concat([df_labels, pd.DataFrame(interpolated_probabilities, columns=[f'prob_{name}' for name in class_names])], axis=1)
        df_final['source_fps'] = source_fps
        df_final.to_csv(output_csv_path, index=False)
        print(f"@@@ SUCCESS: Saved visualization to {output_video_path} @@@")
        print(f"@@@ SUCCESS: Saved CSV labels to {output_csv_path} @@@")

        del all_frames_downsampled, all_yolo_details, behavior_predictions, final_behaviors
        gc.collect()

    print("\n$$$ All videos processed successfully! $$$")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run a full YOLO detection and SlowFast classification pipeline.")
    
    parser.add_argument("--input-path", type=str, required=True, help="Path to input video file or folder.")
    parser.add_argument("--output-folder", type=str, default="full_pipeline_output", help="Folder to save all outputs.")
    parser.add_argument("--yolo-model", type=str, required=True, help="Path to trained YOLOv8 model.")
    parser.add_argument("--slowfast-model", type=str, required=True, help="Path to trained SlowFast model.")
    parser.add_argument("--slowfast-config", type=str, required=True, help="Path to SlowFast YAML config.")
    parser.add_argument("--training-data-path", type=str, required=True, help="Path to the root of training data to infer class names.")
    
    parser.add_argument("--gpu-id", type=int, default=0, help="ID of GPU to use.")
    parser.add_argument("--fps", type=int, default=None, help="Optional: Output FPS for visualization video.")
    parser.add_argument("--downsample", type=float, default=1.0, help="Downsample ratio for processing frames.")
    parser.add_argument("--confidence", type=float, default=0.5, help="YOLO detection confidence threshold.")
    parser.add_argument("--crop-size", type=int, default=224, help="Target size for cropping around the detected mouse.")
    parser.add_argument("--mouse-class-id", type=int, default=0, help="The class ID for 'mouse' in the YOLO model.")
    parser.add_argument("--use-amp", action="store_true", help="Enable Automatic Mixed Precision (AMP) for faster inference.")

    parser.add_argument("--yolo-batch-size", type=int, default=16, help="Batch size for YOLO inference.")
    parser.add_argument("--slowfast-batch-size", type=int, default=8, help="Batch size for SlowFast inference.")
    parser.add_argument("--stride", type=int, default=2, help="Stride for the SlowFast sliding window.")
    parser.add_argument("--sample-stride", type=int, default=1, help="Frame sampling stride within a SlowFast clip.")
    
    parser.add_argument("--min-event-duration", type=int, default=15, help="Minimum frame duration for a behavior event.")
    parser.add_argument("--behavior-thresholds", type=str, default=None, help="Optional: Custom probability thresholds for each class.")
    parser.add_argument("--skip-frames-for-visualization", type=int, default=1, help="Write every Nth frame to the visualization video.")
    parser.add_argument("--codec", type=str, default='mp4v', help="Codec for writing the output video (e.g., 'avc1', 'mp4v').")

    args = parser.parse_args()
    run_pipeline(args)

import os
import json
import argparse
import logging
import re
import yaml
from moviepy.editor import VideoFileClip
from concurrent.futures import ProcessPoolExecutor

import cv2
import numpy as np
import torch
from ultralytics import YOLO
from tqdm import tqdm

DEFAULT_VIDEO_FPS = 30

logging.getLogger("moviepy").setLevel(logging.ERROR)

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def process_video_for_inference(model, input_video_path, output_cropped_path, output_visualized_path,
                                target_size, batch_size, conf_threshold, downsample_ratio, configured_fps):
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        return f"Error: Could not open video '{os.path.basename(input_video_path)}'. Please check the path or video integrity."

    orig_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    detected_fps = cap.get(cv2.CAP_PROP_FPS)
    fps_to_use = detected_fps if detected_fps > 0 else configured_fps

    if fps_to_use <= 0:
        fps_to_use = DEFAULT_VIDEO_FPS
        if detected_fps <= 0 and configured_fps <=0:
             print(f"Warning: Invalid FPS detected for '{os.path.basename(input_video_path)}' (detected: {detected_fps}, configured: {configured_fps}). Defaulting to {fps_to_use} FPS.")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    frame_width = int(orig_width * downsample_ratio)
    frame_height = int(orig_height * downsample_ratio)

    if frame_width <= 0: frame_width = 1
    if frame_height <= 0: frame_height = 1

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    if target_size <= 0:
        cap.release()
        return f"Error: Invalid target_size ({target_size}) for '{os.path.basename(input_video_path)}'. Must be greater than 0."

    out_cropped = cv2.VideoWriter(output_cropped_path, fourcc, fps_to_use, (target_size, target_size))
    if not out_cropped.isOpened():
        cap.release()
        return f"Error: Could not open output video writer for cropped video: '{os.path.basename(output_cropped_path)}'. Check path and permissions."

    out_visualized = None
    if output_visualized_path:
        out_visualized = cv2.VideoWriter(output_visualized_path, fourcc, fps_to_use, (frame_width, frame_height))
        if not out_visualized.isOpened():
            cap.release()
            out_cropped.release()
            return f"Error: Could not open output video writer for visualized video: '{os.path.basename(output_visualized_path)}'. Check path and permissions."

    last_known_box = None
    frame_batch = []

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                if frame_batch:
                    last_known_box = process_inference_batch(
                        model, frame_batch, out_cropped, out_visualized,
                        last_known_box, frame_width, frame_height, target_size, conf_threshold
                    )
                break

            if downsample_ratio != 1.0:
                frame = cv2.resize(frame, (frame_width, frame_height), interpolation=cv2.INTER_AREA)

            frame_batch.append(frame)

            if len(frame_batch) == batch_size:
                last_known_box = process_inference_batch(
                    model, frame_batch, out_cropped, out_visualized,
                    last_known_box, frame_width, frame_height, target_size, conf_threshold
                )
                frame_batch = []

    except Exception as e:
        return f"CRITICAL ERROR processing video '{os.path.basename(input_video_path)}': {e}"
    finally:
        cap.release()
        out_cropped.release()
        if out_visualized:
            out_visualized.release()

    return f"Successfully processed '{os.path.basename(input_video_path)}'."

def process_inference_batch(model, frames, out_cropped, out_visualized, last_known_box, frame_width, frame_height, target_size, conf_threshold):
    if not frames:
        return last_known_box

    results = model(frames, verbose=False, conf=conf_threshold)

    current_batch_last_box = last_known_box

    for i, r in enumerate(results):
        frame = frames[i]
        vis_frame = frame.copy() if out_visualized else None

        mouse_box = None
        target_class_id = 0

        best_conf = -1
        for box in r.boxes:
            if int(box.cls[0]) == target_class_id:
                if box.conf[0] > best_conf:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    x, y, w, h = int(x1), int(y1), int(x2 - x1), int(y2 - y1)
                    mouse_box = (x, y, w, h)
                    best_conf = box.conf[0]

        if mouse_box is None and current_batch_last_box is not None:
            mouse_box = current_batch_last_box
        elif mouse_box is not None:
            current_batch_last_box = mouse_box

        if mouse_box is not None:
            x, y, w, h = mouse_box
            center_x = x + w // 2
            center_y = y + h // 2

            start_x = center_x - target_size // 2
            start_y = center_y - target_size // 2

            cropped_frame = np.zeros((target_size, target_size, 3), dtype=np.uint8)

            frame_start_x = max(0, start_x)
            frame_start_y = max(0, start_y)
            frame_end_x = min(frame_width, start_x + target_size)
            frame_end_y = min(frame_height, start_y + target_size)

            paste_start_x = max(0, -start_x)
            paste_start_y = max(0, -start_y)

            if frame_end_x > frame_start_x and frame_end_y > frame_start_y:
                frame_region = frame[frame_start_y:frame_end_y, frame_start_x:frame_end_x]
                paste_end_x = paste_start_x + frame_region.shape[1]
                paste_end_y = paste_start_y + frame_region.shape[0]
                cropped_frame[paste_start_y:paste_end_y, paste_start_x:paste_end_x] = frame_region

            out_cropped.write(cropped_frame)

            if out_visualized:
                if best_conf != -1:
                    cv2.rectangle(vis_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(vis_frame, f"{r.names[target_class_id]} {best_conf:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

                out_visualized.write(vis_frame)
        else:
            black_cropped_frame = np.zeros((target_size, target_size, 3), dtype=np.uint8)
            out_cropped.write(black_cropped_frame)
            if out_visualized:
                black_vis_frame = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)
                out_visualized.write(black_vis_frame)

    return current_batch_last_box

def init_model_for_worker(model_path, device_id):
    device = torch.device(f'cuda:{device_id}' if torch.cuda.is_available() else 'cpu')
    model = YOLO(model_path)
    model.to(device)
    return model, device

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run inference on videos using a custom-trained YOLOv8 model.")

    parser.add_argument("model_path", type=str, help="Path to your trained YOLOv8 model file.")
    parser.add_argument("input_path", type=str, help="Path to the input video file or a folder containing video files.")
    parser.add_argument("-o", "--output_folder", type=str, default="yolo_inference_videos", help="Folder to save the output videos.")
    parser.add_argument("-s", "--size", type=int, default=112, help="The target size for the cropped video.")
    parser.add_argument("-b", "--batch_size", type=int, default=16, help="Batch size for inference.")
    parser.add_argument("-c", "--confidence", type=float, default=0.5, help="Confidence threshold for detection.")
    parser.add_argument("--gpu_id", type=int, default=0, help="ID of the GPU to use (for single GPU per worker).")
    parser.add_argument("--visualize", action="store_true", help="If set, saves a video with bounding boxes drawn.")
    parser.add_argument("--downsample", type=float, default=1.0, help="Downsample ratio (e.g., 0.5 for half resolution).")
    parser.add_argument("--workers", type=int, default=1, help="Number of parallel CPU processes to use for video processing. Each worker loads the model onto the specified GPU/CPU.")
    parser.add_argument('--config-file', type=str, default='config.yaml', help="Path to the configuration YAML file.")

    args = parser.parse_args()

    try:
        config = load_config(args.config_file)
        args.video_fps = config.get('DATA', {}).get('VIDEO_FPS', DEFAULT_VIDEO_FPS)
        print(f"yayayayay Successfully loaded config from {args.config_file}. Using VIDEO_FPS: {args.video_fps}")
    except FileNotFoundError:
        print(f"ehh.... Error: Config file not found at {args.config_file}. Using default VIDEO_FPS: {DEFAULT_VIDEO_FPS}")
        args.video_fps = DEFAULT_VIDEO_FPS
    except yaml.YAMLError as e:
        print(f"ehh.... Error: YAML decoding failed for {args.config_file}. Please check YAML syntax. Error: {e}. Using default VIDEO_FPS: {DEFAULT_VIDEO_FPS}")
        args.video_fps = DEFAULT_VIDEO_FPS


    video_files_to_process = []
    supported_formats = ('.mp4', '.avi', '.mov', '.mkv')

    if os.path.isfile(args.input_path):
        video_files_to_process.append(args.input_path)
    elif os.path.isdir(args.input_path):
        for root, _, files in os.walk(args.input_path):
            for file in files:
                if file.lower().endswith(supported_formats):
                    video_files_to_process.append(os.path.join(root, file))
    else:
        print(f"Error: Input path '{args.input_path}' is not a valid file or directory. Exiting.")
        exit()

    if not video_files_to_process:
        print(f"No video files found at '{args.input_path}'. Exiting.")
        exit()

    video_files_to_process.sort()

    tasks = []
    for video_path in video_files_to_process:
        relative_path = os.path.relpath(video_path, args.input_path)
        base, ext = os.path.splitext(relative_path)

        output_cropped_path_full = os.path.join(args.output_folder, "cropped", f"{base}_cropped{ext}")
        os.makedirs(os.path.dirname(output_cropped_path_full), exist_ok=True)

        output_visualized_path_full = None
        if args.visualize:
            output_visualized_path_full = os.path.join(args.output_folder, "visualized", f"{base}_visualized{ext}")
            os.makedirs(os.path.dirname(output_visualized_path_full), exist_ok=True)

        tasks.append((
            args.model_path,
            args.gpu_id,
            video_path,
            output_cropped_path_full,
            output_visualized_path_full,
            args.size,
            args.batch_size,
            args.confidence,
            args.downsample,
            args.video_fps
        ))

    print(f"🚀 Starting inference for {len(tasks)} videos...")

    results = []
    if args.workers == 1:
        device = torch.device(f'cuda:{args.gpu_id}' if torch.cuda.is_available() else 'cpu')
        model = YOLO(args.model_path)
        model.to(device)
        print(f"Running in single-process mode. Using device: {device}")

        for task_args in tqdm(tasks, desc="Overall Progress"):
            res = process_video_for_inference(
                model,
                task_args[2],
                task_args[3],
                task_args[4],
                task_args[5],
                task_args[6],
                task_args[7],
                task_args[8],
                task_args[9]
            )
            results.append(res)

    else:
        print(f"Running in parallel with {args.workers} workers. Each worker will initialize its own model on GPU {args.gpu_id} (if available).")
        def worker_process_wrapper(task_args):
            worker_model, worker_device = init_model_for_worker(task_args[0], task_args[1])
            return process_video_for_inference(
                worker_model,
                task_args[2],
                task_args[3],
                task_args[4],
                task_args[5],
                task_args[6],
                task_args[7],
                task_args[8],
                task_args[9]
            )

        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            results = list(tqdm(executor.map(worker_process_wrapper, tasks), total=len(tasks), desc="Overall Progress"))

    print("\n--- Processing Summary ---")
    for res in results:
        if "Error" in res or "CRITICAL ERROR" in res:
            print(f"ehh.... {res}")
        else:
            print(f"good {res}")
    print("\nAll inference tasks complete.")

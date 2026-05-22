import os
import json
import argparse
import logging
import re
import yaml
from moviepy.editor import VideoFileClip
from concurrent.futures import ProcessPoolExecutor
import tqdm

DEFAULT_CLIP_LENGTH = 32
DEFAULT_MIN_LENGTH = 16 
DEFAULT_TRAIN_STRIDE = 16
DEFAULT_SAMPLE_STRIDE = 1
DEFAULT_VIDEO_FPS = 30
DEFAULT_BEHAVIOR_FOLDER = "Etc"

logging.getLogger("moviepy").setLevel(logging.ERROR)

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def get_mouse_id(filename):
    basename = os.path.basename(filename).lower()
    match = re.search(r'm\d{2}', basename)
    if match:
        return match.group(0)
    match_fst = re.search(r'(fst_d\d_m\d{2})', basename)
    if match_fst:
        return match_fst.group(1) 
    return None

def get_behavior_from_filename(filename, known_behaviors):
    sorted_behaviors = sorted(known_behaviors, key=len, reverse=True)
    
    filename_lower = filename.lower()

    for behavior in sorted_behaviors:
        pattern = r'[\W_]' + re.escape(behavior.lower()) + r'[\W_]'
        if re.search(pattern, filename_lower):
            return behavior
        if filename_lower.startswith(behavior.lower() + '_') or \
           filename_lower.endswith('_' + behavior.lower()):
            return behavior
    
    return DEFAULT_BEHAVIOR_FOLDER

def process_video_segment(args_tuple):
    source_video_path, is_test_video, script_args = args_tuple
    disk_filename = os.path.basename(source_video_path)
    base_name_without_ext = os.path.splitext(disk_filename)[0]
    
    output_base_dir = os.path.join(script_args.output_dir, 'test') if is_test_video else os.path.join(script_args.output_dir, 'train')
    
    behavior = get_behavior_from_filename(disk_filename, script_args.all_behaviors)
    
    behavior_dir = os.path.join(output_base_dir, behavior)
    os.makedirs(behavior_dir, exist_ok=True) 
    
    current_stride = script_args.stride
    if is_test_video:
        current_stride = script_args.clip_length

    configured_frame_rate = script_args.video_fps 
    total_clips_saved = 0
    
    base_ffmpeg_params = ['-c:v', 'h264_nvenc'] if script_args.use_gpu else []
    moviepy_logger = None 

    try:
        with VideoFileClip(source_video_path, fps_source="fps") as main_clip:
            clip_duration_seconds = script_args.clip_length / configured_frame_rate
            
            total_frames_in_input_video = int(main_clip.duration * configured_frame_rate) 
            
            output_fps_for_write = configured_frame_rate / script_args.sample_stride

            possible_starts = []
            
            current_start_frame = 0
            while (current_start_frame + script_args.clip_length) <= total_frames_in_input_video:
                possible_starts.append(current_start_frame)
                current_start_frame += current_stride
            
            if total_frames_in_input_video >= script_args.min_length:
                last_clip_start_frame = max(0, total_frames_in_input_video - script_args.clip_length)
                
                actual_last_clip_length = total_frames_in_input_video - last_clip_start_frame
                
                if actual_last_clip_length >= script_args.min_length:
                    if not possible_starts or \
                       (last_clip_start_frame > possible_starts[-1] and \
                        abs(last_clip_start_frame - possible_starts[-1]) >= current_stride): 
                        possible_starts.append(last_clip_start_frame)
                        possible_starts.sort() 

            for clip_idx, start_f in enumerate(possible_starts):
                start_time_sec = start_f / configured_frame_rate
                end_time_sec = min(start_time_sec + clip_duration_seconds, main_clip.duration)

                if end_time_sec <= start_time_sec:
                    logging.warning(f"Skipping clip for {disk_filename} - Invalid time range: [{start_time_sec}, {end_time_sec}]")
                    continue
                
                extracted_clip_frames = int((end_time_sec - start_time_sec) * configured_frame_rate)
                if extracted_clip_frames < script_args.min_length:
                    logging.info(f"Skipping clip for {disk_filename} at start_frame {start_f} due to min_length ({extracted_clip_frames} < {script_args.min_length}).")
                    continue

                subclip = main_clip.subclip(start_time_sec, end_time_sec)
                
                output_clip_filename = f"{base_name_without_ext}_resampled_{clip_idx:04d}.mp4"
                output_clip_path = os.path.join(behavior_dir, output_clip_filename)

                subclip.write_videofile(
                    output_clip_path, 
                    codec="libx264", 
                    audio_codec="aac", 
                    logger=moviepy_logger, 
                    ffmpeg_params=base_ffmpeg_params, 
                    fps=output_fps_for_write
                )
                total_clips_saved += 1

        return f"Processed: {disk_filename} -> Resampled {total_clips_saved} clips into {behavior}"

    except Exception as e:
        return f"CRITICAL ERROR on {disk_filename}: {e}"

def main():
    parser = argparse.ArgumentParser(
        description="Resample pre-cropped mouse behavior video clips to constant length and classify into train/test sets and behavior-specific folders.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--input-dir', type=str, required=True, help="Directory containing pre-cropped video files (and their subfolders).")
    parser.add_argument('--json-file', type=str, required=True, help="Path to the JSON file containing 'behaviors' list (e.g., NewTubba.json).") 
    parser.add_argument('--config-file', type=str, default='config.yaml', help="Path to the configuration YAML file.")
    parser.add_argument('--output-dir', type=str, default='resampled_clips', help="Directory where 'train' and 'test' folders will be created.") 
    parser.add_argument('--test-videos', type=str, nargs='+', required=False, default=[], 
                        help="List of mouse IDs (e.g., m02 m45) for the test set. Videos containing these IDs will be assigned to the test set.")
    parser.add_argument('--workers', type=int, default=os.cpu_count(), help="Number of parallel CPU processes to use.")
    parser.add_argument('--use-gpu', action='store_true', help="Enable GPU-accelerated video encoding (requires NVIDIA GPU and compatible ffmpeg).") 
    parser.add_argument('--clip-length', type=int, default=DEFAULT_CLIP_LENGTH, help="Desired length of each output clip in frames.")
    parser.add_argument('--min-length', type=int, default=DEFAULT_MIN_LENGTH, 
                        help="Minimum length of an output clip in frames. Partial clips at the end of an input video will be skipped if shorter than this.")
    parser.add_argument('--stride', type=int, default=DEFAULT_TRAIN_STRIDE, help="Sliding window stride for re-sampling (in frames). This will be used for training set. For test set, clip_length will be used to prevent overlap.")
    parser.add_argument('--no-parallel', action='store_true', help="Disable parallel processing for debugging.")
    parser.add_argument('--verbose', action='store_true', help="Enable verbose FFMPEG logs for debugging.")
    parser.add_argument('--sample-stride', type=int, default=DEFAULT_SAMPLE_STRIDE, help="Output every Nth frame by reducing output FPS. A value of 1 means no sampling.")


    args = parser.parse_args()

    try:
        config = load_config(args.config_file)
        args.video_fps = config.get('DATA', {}).get('VIDEO_FPS', DEFAULT_VIDEO_FPS)
        print(f"@@@ SUCCESS: Successfully loaded config from {args.config_file}. Using VIDEO_FPS: {args.video_fps} @@@")
    except FileNotFoundError:
        print(f"!!! ERROR: Config file not found at {args.config_file}. Using default VIDEO_FPS: {DEFAULT_VIDEO_FPS} !!!")
        args.video_fps = DEFAULT_VIDEO_FPS
    except yaml.YAMLError as e:
        print(f"!!! ERROR: YAML decoding failed for {args.config_file}. Please check YAML syntax. Error: {e}. Using default VIDEO_FPS: {DEFAULT_VIDEO_FPS} !!!")
        args.video_fps = DEFAULT_VIDEO_FPS

    try:
        with open(args.json_file, 'r') as f:
            data = json.load(f)
        print(f"@@@ SUCCESS: Successfully loaded behaviors from {args.json_file} @@@")
    except FileNotFoundError:
        print(f"!!! ERROR: JSON file not found at {args.json_file} !!!")
        return
    except json.JSONDecodeError as e:
        print(f"!!! ERROR: JSON decoding failed for {args.json_file}. Please check JSON syntax. Error: {e} !!!")
        return

    args.all_behaviors = data.get("behaviors", [])
    if not args.all_behaviors:
        print("!!! WARNING: No 'behaviors' found in the JSON file. Using default 'Etc' folder for all clips. !!!")
        args.all_behaviors = [DEFAULT_BEHAVIOR_FOLDER]

    train_base_dir = os.path.join(args.output_dir, 'train')
    test_base_dir = os.path.join(args.output_dir, 'test')
    
    for behavior in args.all_behaviors:
        os.makedirs(os.path.join(train_base_dir, behavior), exist_ok=True)
        os.makedirs(os.path.join(test_base_dir, behavior), exist_ok=True)
    
    os.makedirs(os.path.join(train_base_dir, DEFAULT_BEHAVIOR_FOLDER), exist_ok=True)
    os.makedirs(os.path.join(test_base_dir, DEFAULT_BEHAVIOR_FOLDER), exist_ok=True)


    tasks = []
    test_set_identifiers = set(args.test_videos)

    disk_videos_found = []
    for root, _, files in os.walk(args.input_dir):
        for file in files:
            if file.lower().endswith('.mp4'):
                disk_videos_found.append(os.path.join(root, file))

    train_set_assigned, test_set_assigned = [], []

    for source_video_path in disk_videos_found:
        disk_filename = os.path.basename(source_video_path)
        video_identifier = get_mouse_id(disk_filename)
        
        is_test = False
        if video_identifier and video_identifier in test_set_identifiers:
            is_test = True
        
        tasks.append((source_video_path, is_test, args)) 
        
        if is_test:
            test_set_assigned.append(disk_filename)
        else:
            train_set_assigned.append(disk_filename)

    if not tasks:
        print("\n!!! ERROR: No videos to process. Exiting. Ensure .mp4 files exist in --input-dir or its subfolders. !!!")
        return

    print(f"### STARTING: Re-sampling and classification for {len(tasks)} videos... ###")

    results = []
    if args.no_parallel:
        print("### INFO: Running in single-process mode for debugging. ###")
        for task in tqdm.tqdm(tasks, total=len(tasks)):
            results.append(process_video_segment(task))
    else:
        print(f"### INFO: Running in parallel with {args.workers} workers... ###")
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            results = list(tqdm.tqdm(executor.map(process_video_segment, tasks), total=len(tasks)))

    print("\n--- Processing Summary ---")
    final_clip_count = 0
    for res in results:
        if "CRITICAL ERROR" in res:
            print(f"!!! ERROR: {res} !!!")
        elif "Resampled" in res:
            print(res)
            try:
                match = re.search(r'Resampled (\d+) clips', res)
                if match:
                    count = int(match.group(1))
                    final_clip_count += count
            except (AttributeError, ValueError):
                pass
    
    for tv_id_arg in args.test_videos:
        if not any(get_mouse_id(fn) == tv_id_arg for fn in test_set_assigned):
            print(f"!!! WARNING: Test video ID '{tv_id_arg}' was specified but no corresponding video on disk was found or successfully assigned to the test set. !!!")

    print(f"\n$$$ Re-sampling and classification complete! A total of {final_clip_count} clips were created in '{args.output_dir}'. $$$")

if __name__ == "__main__":
    main()

import os
import json
import argparse
from collections import defaultdict
import cv2 
import random 

def get_behavior_frame_counts(json_file_path: str):
    if not os.path.exists(json_file_path):
        print(f"Error: JSON file not found at '{json_file_path}'")
        return {}

    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from '{json_file_path}'. Check file format.")
        return {}
    except Exception as e:
        print(f"An unexpected error occurred while reading '{json_file_path}': {e}")
        return {}

    video_behavior_frames = {}

    for video_data in data.get('videos', []):
        video_name = video_data.get('name')
        if not video_name:
            continue

        annotations = video_data.get('annotations', {})
        behavior_frame_counts = defaultdict(int)

        for behavior_type, segments in annotations.items():
            for start_frame, end_frame, _ in segments:
                behavior_frame_counts[behavior_type] += (end_frame - start_frame + 1)
        
        video_behavior_frames[video_name] = dict(behavior_frame_counts)
    
    return video_behavior_frames

def count_frames_manually(video_path: str):

    if not os.path.exists(video_path):
        print(f"Error: Video file not found at '{video_path}'")
        return None

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file '{video_path}'")
        return None

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
    
    cap.release()
    return frame_count


def extract_random_frames(video_dir: str, num_frames_to_extract: int, output_dir: str):
   
    if not os.path.isdir(video_dir):
        print(f"@@@ Warning: Video directory not found at '{video_dir}'")
        return

    os.makedirs(output_dir, exist_ok=True)
    print(f"@@@ Important: Saving extracted frames to '{output_dir}'")

    video_files = [
        os.path.join(video_dir, f)
        for f in os.listdir(video_dir)
        if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))
    ]

    if not video_files:
        print(f"@@@ Warning: No video files found in '{video_dir}'")
        return

    all_frames = []
    print("\n## Checkpoint: Counting total frames in all videos...")
    for video_path in video_files:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"@@@ Warning: Could not open video '{video_path}'. Skipping.")
            continue
        
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if frame_count > 0:
            all_frames.extend([(video_path, i) for i in range(frame_count)])
            print(f"### Checkpoint: Video '{os.path.basename(video_path)}' has {frame_count} frames.")
        cap.release()

    if not all_frames:
        print("@@@ Warning: No frames were found across any videos. Exiting.")
        return

    num_total_available_frames = len(all_frames)
    if num_frames_to_extract > num_total_available_frames:
        print(f"@@@ Warning: Requested {num_frames_to_extract} frames, but only {num_total_available_frames} are available. Extracting all available frames.")
        num_frames_to_extract = num_total_available_frames

    print(f"\n## Checkpoint: Randomly selecting {num_frames_to_extract} frames...")
    random_frames_to_extract = random.sample(all_frames, num_frames_to_extract)

    saved_count = 0
    for video_path, frame_index in random_frames_to_extract:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"@@@ Warning: Could not reopen video '{video_path}'. Skipping frame.")
            continue
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ret, frame = cap.read()
        cap.release()

        if ret:
            video_basename = os.path.splitext(os.path.basename(video_path))[0]
            output_filename = f"{video_basename}_frame_{frame_index:06d}.jpg"
            output_path = os.path.join(output_dir, output_filename)
            cv2.imwrite(output_path, frame)
            saved_count += 1
    
    print(f"\n## Checkpoint: Finished extracting. Successfully saved {saved_count} frames to '{output_dir}'.")
    if saved_count < num_frames_to_extract:
        print("@@@ Warning: Some frames could not be extracted or saved. Check for potential video corruption.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Helper functions for video analysis based on annotation files."
    )
    parser.add_argument(
        "--behav_frames",
        type=str,
        help="Path to the JSON annotation file to get behavior frame counts."
    )
    parser.add_argument(
        "--numberFrames",
        type=str,
        help="Path to a video file to count its total frames manually."
    )
    parser.add_argument(
        "--extract_frames_yolo",
        type=str,
        help="Path to a directory containing videos to extract random frames from."
    )
    parser.add_argument(
        "--num_frames",
        type=int,
        default=100,
        help="The total number of frames to extract for YOLO training. Requires --extract_frames_yolo."
    )

    args = parser.parse_args()

  
    if args.behav_frames:
        print(f"Calculating behavior frame counts from: {args.behav_frames}")
        frame_counts = get_behavior_frame_counts(args.behav_frames)
        if frame_counts:
            for video, behaviors in frame_counts.items():
                print(f"\nVideo: {video}")
                for behav_type, count in behaviors.items():
                    print(f"  {behav_type}: {count} frames")
        else:
            print("No behavior frame counts could be retrieved.")

  
    if args.numberFrames:
        print(f"\nCounting frames manually for: {args.numberFrames}")
        total_frames = count_frames_manually(args.numberFrames)
        if total_frames is not None:
            print(f"Total frames counted: {total_frames}")

   
    if args.extract_frames_yolo:
        print("\n#####################################################")
        print("### Starting random frame extraction for YOLO...  ###")
        print("#####################################################")
        extract_random_frames(
            video_dir=args.extract_frames_yolo,
            num_frames_to_extract=args.num_frames,
            output_dir="yolo_training_data"
        )

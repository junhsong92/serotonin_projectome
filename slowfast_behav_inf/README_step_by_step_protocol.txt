Hi, I hope this helps your research:) If you have any issues, please drop me an email jun.h.song92@gmail.com.
I'm assuming your are working with an NVIDIA GPU with VRAM > 12GB

Installation (assuming you have conda installed)
1. (change your current directory to the downloaded folder)
2. conda env create -f environment.yml
3. conda activate video_behav_inference_env

Training data prep
1. YOLO annotation for bounding box cropping:
        a. python helper_function.py --extract_frames_yolo YOUR_FOLDER_DIR_WITH_VIDEO_FILES --num_frames 300 (YOLO is robust, so 300 worked well in my case, but you may increase this number to generate more training data)
        b. manual annotation using external tools, e.g. app.roboflow.com (When downloading your (augmented) labeled samples, choose the YOLOv8 format)
        c. unzip
        
2. Behavior annotation
        a. I used https://github.com/tshindmarsh/TUBBA (but if you have something your prefer, please feel free to use them; whatever works works, as long as the formats match.)
        b. when you label for training, try to sample more from "hard" frame/windows, e.g. behavior transitions
        b. You will have a json file with behavior labels.
        
        
YOLO Training: this should take only a few hours. you can adjust the batch size (-b) based on your VRAM size
1. python train_yolo.py YOUR_ROBOFLOW_DOWNLOADED_FOLDER/data.yaml -m yolov8s.pt -e 100 -b 128 --name mouse_yolov8s_50epochs
2. python inference_yolo_detection.py YOUR_DIR/weights/best.pt YOUR_DIR/tubba_chunks --size 224 --batch_size 64 --confidence 0.5 --downsample 0.25 --gpu_id 0 --workers 8 -o yolo_cropped ###

SLOWFAST Training (adjust the batch size and num_workers):

1. python prepare_slowfast_traintest_set.py --input-dir YOUR_DIR/yolo_cropped/cropped --json-file behav_label.json --output-dir slowfast_training_data   --verbose --clip-length 32 --min-length 1 --stride 2 --sample-stride 1 --workers 32 --test-videos SAMPLES_YOU_WANT_TO_USE_FOR_TESTING
2. python train_slowfast2.py --config config.yaml --data-path YOUR_FOLDER/slowfast_training_data --output-dir slowfast_model --new-lr 1e-4 --warmup-epochs 5  --new-batch-size 96 --fps 30 --confusion-off-diag-vis --lr-backbone 2e-5 --patience 15 --min-delta 0.001 --label-smooth 0.1 --use-amp --lr-patience 3 --stoch-sampling-percent 1 
3. (one or two days later....) Training might take long

INFERENCE:
1. run_full_inference.py --input-path RAW_VIDEO_PATH --yolo-model YOUR_YOLO_DIR/weights/best.pt --slowfast-model YOUR_SLOWFAST_DIR/best_model.pth --slowfast-config config.yaml --training-data-path slowfast_training_data --fps 30 --stride 4 --min-event-duration 12 --yolo-batch-size 64 --downsample 0.5 --use-amp
2. (Go home and grab some beers)
3. (next morning) tada!
        

        

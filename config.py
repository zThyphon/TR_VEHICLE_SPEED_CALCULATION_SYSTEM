config = {
    "INPUT_VIDEO_PATH": "traffic.mp4",              # define input video path (assign 0 for connecting default camera)
    "VEHICLE_MODEL_PATH": "./models/yolov8s.pt",               # path to vehicle detection model
    "PLATE_MODEL_PATH": "./models/license_plate_detector.pt",  # path to license plate model
    "PLATE_OUTPUT_DIR": "plate_output",             # directory for saving plate images
    "LOG_FILE": "overspeed_log.txt",           # file path for speed logging
    "CONF": 0.25,                             # detection confidence threshold
    "VEHICLE_CLASSES": [ # vehicle class IDs (COCO)
        2, # car
        3, # motorcycle
        5, # bus
        7 # truck 
    ],           


    # ============================ FRAME CONFIGS ============================

    "FRAME_DIMENSIONS": (960, 540),            # display frame resolution
    "SHOW_FRAME": True,                       # enable or disable video display

    # ============================ QUEUE CONFIGS ============================
    "PLATE_QUEUE_MAX_SIZE": 128, # define max queue size for plate detection
    "DISPLAY_QUEUE_MAX_SIZE": 5, # define max queue size for plate detection

    # ============================ SPEED CONFIGS ============================

    "METERS_PER_PIXEL": 0.1,                  # conversion factor from pixels to meters
    "REF_BOX_HEIGHT": 160,                    # reference box height for depth correction
    "MAX_REALISTIC_SPEED": 200,               # maximum allowed speed (km/h)
    "EMA_ALPHA": 0.25,                        # EMA smoothing coefficient
    "SPEED_LIMIT": 10,                        # speed threshold


    # ============================ PLATE CONFIGS ============================

    "MIN_CONF_OCR": 0.6,                      # minimum OCR confidence threshold
    "MIN_SHARPNESS": 40.0,                    # minimum sharpness for OCR
    "MAX_PLATE_ATTEMPTS": 5,                  # maximum OCR attempts per vehicle
    "PLATE_EXPAND_SCALE": 0.35,               # expansion scale for vehicle crop
    "FINAL_PLATE_EXPAND": 0.20,               # expansion scale for plate crop
    "OCR_RETRY_GAP": 10,                      # frames between OCR retries


    # ============================ TRACKING CONFIGS ============================

    "TRACK_TTL_FRAMES": 150,                  # frames before a track expires
    "TRACK_EVERY_N": 3,                       # process every N frames


    # ============================ OPTICAL FLOW CONFIGS ============================

    "FLOW_EVERY_N": 5,     # compute optical flow every N frames (for hardware optimization)
}
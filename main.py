import cv2
import threading
import queue
import numpy as np
from ultralytics import YOLO
from config import config
from utils import expand_box, create_plate_output_dirs
from speed import compute_speed, cleanup_tracks
from thread_workers import plate_worker, display_worker


def main():
    # ===================== MODELS =====================

    # load YOLO vehicle detection and tracking model
    vehicle_model = YOLO(config["VEHICLE_MODEL_PATH"])

    # load YOLO license plate detection model
    plate_model = YOLO(config["PLATE_MODEL_PATH"])

    # fuse model layers for faster inference
    vehicle_model.fuse()
    plate_model.fuse()

    # ===================== STATES =====================

    # dictionary to store tracking-related state (speed, last position, etc.)
    track_state = {}

    # dictionary to store plate OCR state per track ID
    plate_state = {}

    # cache last known bounding boxes per track ID
    last_boxes = {}

    # cache last known speeds per track ID
    last_speeds = {}

    # queue used to send cropped vehicle images to OCR worker thread
    plate_queue = queue.Queue(maxsize=config["PLATE_QUEUE_MAX_SIZE"])

    # queue used to send frames to the display thread
    display_queue = queue.Queue(maxsize=config["DISPLAY_QUEUE_MAX_SIZE"])

    # event used to signal display thread to stop
    stop_display = threading.Event()

    # create directories where plate images will be saved
    create_plate_output_dirs()

    # ===================== THREADS =====================

    # start OCR worker thread (for plate detection)
    threading.Thread(
        target=plate_worker,
        args=(plate_queue, plate_state, plate_model),
        daemon=True  # Daemon thread exits when main thread exits
    ).start()

    # start display worker thread
    threading.Thread(
        target=display_worker,
        args=(display_queue, stop_display),
        daemon=True
    ).start()

    # ===================== VIDEO =====================

    # open the input video
    cap = cv2.VideoCapture(config["INPUT_VIDEO_PATH"])

    # read FPS from video
    fps = cap.get(cv2.CAP_PROP_FPS) or 30

    # frame counter
    frame_idx = 0

    # store previous grayscale frame for optical flow
    prev_gray = None

    # store last affine transform matrix from optical flow
    last_flow_M = None

    # ===================== MAIN LOOP =====================

    # loop until video ends or display thread requests stop
    while cap.isOpened() and not stop_display.is_set():
        
        # read next video frame
        ret, frame = cap.read()
        if not ret:
            break  # exit loop if video ends

        # increment frame counter
        frame_idx += 1

        # keep a clean copy of the frame for cropping
        clean_frame = frame.copy()

        # convert frame to grayscale (required for optical flow)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # ---------- Optical Flow ----------
        # compute optical flow every N frames
        if prev_gray is not None and frame_idx % config["FLOW_EVERY_N"] == 0:
            # Compute dense optical flow between frames
            flow = cv2.calcOpticalFlowFarneback(
                prev_gray, gray, None,
                0.5, 3, 15, 3, 5, 1.2, 0
            )

            # compute average motion in x and y direction
            dx, dy = flow[..., 0].mean(), flow[..., 1].mean()

            # create affine transform to stabilize frame
            last_flow_M = np.float32([[1, 0, -dx], [0, 1, -dy]])

        # apply motion compensation if available
        if last_flow_M is not None:
            frame = cv2.warpAffine(frame, last_flow_M, frame.shape[1::-1])
            clean_frame = cv2.warpAffine(clean_frame, last_flow_M, clean_frame.shape[1::-1])

        # update previous grayscale frame
        prev_gray = gray

        # ---------- YOLO TRACK ----------
        # run YOLO tracking only every N frames (performance optimization)
        if frame_idx % config["TRACK_EVERY_N"] == 0:
            results = vehicle_model.track(
                frame,
                persist=True,                       # Keep track IDs persistent
                conf=config["CONF"],               # Confidence threshold
                classes=config["VEHICLE_CLASSES"], # Vehicle classes only
                verbose=False
            )[0]

            # check if detections and IDs exist
            if results.boxes and results.boxes.id is not None:
                for box, tid in zip(
                    results.boxes.xyxy.cpu().numpy(),
                    results.boxes.id.cpu().numpy()
                ):
                    # convert track ID to integer
                    tid = int(tid)

                    # extract bounding box coordinates
                    x1, y1, x2, y2 = map(int, box)

                    # compute vehicle speed
                    speed = compute_speed(
                        track_state,
                        tid,
                        y2,                      # bottom of bounding box
                        y2 - y1,                 # box height
                        frame_idx,
                        fps,
                        REF_BOX_HEIGHT=config["REF_BOX_HEIGHT"],
                        EMA_ALPHA=config["EMA_ALPHA"],
                        METERS_PER_PIXEL=config["METERS_PER_PIXEL"],
                        MAX_REALISTIC_SPEED=config["MAX_REALISTIC_SPEED"],
                    )

                    # save latest bounding box and speed
                    last_boxes[tid] = (x1, y1, x2, y2)
                    last_speeds[tid] = speed

                    # initialize plate OCR state if needed
                    ps = plate_state.setdefault(
                        tid, {"done": False, "attempts": 0, "last_try": 0}
                    )

                    # schedule OCR if conditions are met
                    if (
                        not ps["done"]
                        and ps["attempts"] < config["MAX_PLATE_ATTEMPTS"]
                        and frame_idx - ps["last_try"] > config["OCR_RETRY_GAP"]
                    ):
                        ps["attempts"] += 1
                        ps["last_try"] = frame_idx

                        # expand bounding box for better plate visibility
                        fx1, fy1, fx2, fy2 = expand_box(
                            x1, y1, x2, y2,
                            frame.shape[1], frame.shape[0],
                            config["PLATE_EXPAND_SCALE"]
                        )

                        # crop vehicle image
                        crop = clean_frame[fy1:fy2, fx1:fx2].copy()

                        # send crop to OCR worker thread (non-blocking)
                        try:
                            plate_queue.put_nowait((tid, crop, speed))
                        except queue.Full:
                            pass

        # ---------- DRAW EVERY FRAME ----------

        # draw cached boxes even on frames where YOLO is skipped
        for tid, (x1, y1, x2, y2) in last_boxes.items():
            speed = last_speeds.get(tid, 0)

            # color box based on speed limit (red and green)
            color = (0, 0, 255) if speed > config["SPEED_LIMIT"] else (0, 255, 0)

            # draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # draw speed text
            cv2.putText(
                frame,
                f"{int(speed)} km/h",
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2
            )

        # ---------- CLEANUP ----------

        # remove expired tracks
        expired = cleanup_tracks(track_state, frame_idx, config["TRACK_TTL_FRAMES"])
        for tid in expired:
            track_state.pop(tid, None)
            plate_state.pop(tid, None)
            last_boxes.pop(tid, None)
            last_speeds.pop(tid, None)

        # ---------- DISPLAY ----------
        
        # send frame to display thread (drop if queue is full)
        try:
            display_queue.put_nowait(frame.copy())
        except queue.Full:
            pass

    # ===================== CLEAN SHUTDOWN =====================

    # release video resource
    cap.release()

    # signal display thread to stop
    stop_display.set()

    # send sentinel to display queue
    try:
        display_queue.put_nowait(None)
    except queue.Full:
        pass

    # stop OCR worker thread
    
    plate_queue.put(None)
    plate_queue.join()

if __name__ == "__main__":
    main()
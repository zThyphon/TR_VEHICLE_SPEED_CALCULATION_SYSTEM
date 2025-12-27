import cv2
import os
import queue
from ocr import use_ocr
from utils import expand_box
from log import log_plate
from config import config

""" 
This worker runs in a separate thread and performs:
    - license plate detection
    - OCR
    - saving plate images
    - logging results
"""
def plate_worker(plate_queue, plate_state, plate_model):

    # run indefinitely until a stop signal (None) is received
    while True:

        # get next item from the plate queue (blocks until available)
        item = plate_queue.get()

        try:
            # if sentinel value is received, exit the worker loop
            if item is None:
                break
            
            """
                Unpack queue item:
                tid   → tracking ID
                vehicle_crop → cropped image of the vehicle
                speed → computed vehicle speed
            """
            tid, vehicle_crop, speed = item

            # retrieve plate processing state for this track ID
            ps = plate_state.get(tid)

            """
                skip processing if:
                - no state exists
                - plate has already been successfully processed
            """
            if not ps or ps["done"]:
                continue

            # run license plate detection model on the vehicle crop
            res = plate_model(vehicle_crop, conf=0.5, verbose=False)[0]

            # if no plate bounding boxes were detected, skip
            if not res.boxes:
                continue

            # get height and width of the vehicle crop
            h, w = vehicle_crop.shape[:2]

            # iterate over detected plate bounding boxes
            for b in res.boxes.xyxy.cpu().numpy():

                # convert bounding box coordinates to integers
                x1, y1, x2, y2 = map(int, b)

                # expand the plate bounding box for better OCR accuracy
                px1, py1, px2, py2 = expand_box(
                    # original plate box
                    x1, y1, x2, y2, 
                    # image dimensions
                    w, h,            
                    # expansion scale
                    config["FINAL_PLATE_EXPAND"] 
                )

                # crop the license plate region from the vehicle image
                crop = vehicle_crop[py1:py2, px1:px2]

                # attempt OCR on the cropped plate image
                plate_text = use_ocr(crop)

                # if OCR successfully returns text
                if plate_text:
                    # save cropped license plate image
                    cv2.imwrite(
                        os.path.join(
                            config["PLATE_OUTPUT_DIR"],
                            f"{plate_text}.jpg"
                        ),
                        crop
                    )

                    # log detected plate number and associated speed
                    log_plate(plate_text, speed)

                    # print detected overspeed plate to console
                    print(f"Overspeed Plate: {plate_text}")

                    # mark this track ID as completed to avoid reprocessing
                    ps["done"] = True

                    # stop checking other plate detections for this vehicle
                    break

        finally:
            # mark the queue task as done (important to avoid deadlocks)
            plate_queue.task_done()


# this worker handles all OpenCV GUI operations in a single thread
def display_worker(display_queue, stop_display):
    # run display loop until stop signal is set
    while not stop_display.is_set():

        try:
            # attempt to get a frame from the display queue
            frame = display_queue.get(timeout=0.1)

        # if no frame is available yet, continue looping
        except queue.Empty:
            continue

        # if sentinel value is received, exit display loop
        if frame is None:
            break

        # display the resized frame in the window
        cv2.imshow(
            "TR VEHICLE SPEED CALCULATION SYSTEM",
            cv2.resize(frame, config["FRAME_DIMENSIONS"])
        )

        # wait for a key press (1 ms)
        # exit if 'q' or ESC is pressed
        if cv2.waitKey(1) & 0xFF in (27, ord("q")):
            stop_display.set()
            break

    # close all OpenCV windows safely (must be called from same thread)
    cv2.destroyAllWindows()
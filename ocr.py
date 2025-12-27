import cv2
from paddleocr import PaddleOCR
from utils import get_sharpness, enhance_plate, clean_text, is_valid_plate_format
from config import config

# get min OCR confidence threshold and minimum sharapness from config
MIN_CONF_OCR = config["MIN_CONF_OCR"]               
MIN_SHARPNESS = config["MIN_SHARPNESS"]

# initialize OCR reader
reader = PaddleOCR(use_angle_cls=True, lang="en")

# function to perform OCR on plate crop
def use_ocr(plate_crop):                       

    # check if crop is valid or not
    if plate_crop is None or plate_crop.size == 0:  
        return None                      
    # check sharpness of the cropped plates (reject blurry images)
    if get_sharpness(plate_crop) < MIN_SHARPNESS:
        return None

    # get crop height and width
    h, w = plate_crop.shape[:2]
    # check if plate is too small
    if w < 150:            
         # apply upscaling
        plate_crop = cv2.resize(plate_crop, (w * 2, h * 2), interpolation=cv2.INTER_LANCZOS4) 

    # enhance the plate
    plate_crop = enhance_plate(plate_crop)
    # run ocr
    result = reader.predict(plate_crop)
    
    # check OCR returns something or nots
    if not result:                          
        return None                   

    # get best OCR result
    best_text, best_conf = None, 0   
    # get OCR results
    for text, conf in zip(result[0]["rec_texts"], result[0]["rec_scores"]):
        # clean OCR text
        text = clean_text(text)
        if len(text) >= 5 and conf >= MIN_CONF_OCR and conf > best_conf and is_valid_plate_format(text):
            # update best match
            best_text, best_conf = text, conf

    # return detected plate text
    return best_text
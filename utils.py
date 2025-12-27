import cv2
import re
import os
from config import config

PLATE_OUTPUT_DIR = config["PLATE_OUTPUT_DIR"]

# create output directory if it does not exit
def create_plate_output_dirs():
    os.makedirs(PLATE_OUTPUT_DIR, exist_ok=True)

# function to clean OCR text
def clean_text(text):                       
    # keep only alphanumeric characters
    return "".join(char for char in text.upper() if char.isalnum())

# function to validate TR plate format
def is_valid_plate_format(text):       
    # match regex (NN-L(L)(L)(L)-NN(N)(N))
    matched = re.fullmatch(r"([0-9]{2})([A-Z]{1,3})([0-9]{2,4})", text)
    # Validate state code range (1-81) 
    return bool(matched and 1 <= int(matched.group(1)) <= 81)  

# function to compute image sharpness
def get_sharpness(img):                  
    # compute Laplacian
    variance = cv2.Laplacian(                
        # convert to grayscale
        cv2.cvtColor(img, cv2.COLOR_BGR2GRAY),
        # use 64-bit float depth 
        cv2.CV_64F          
    ).var()          

    # return variance as sharpnesss 
    return variance                    

# function to enhance plate image
def enhance_plate(img):                  
    # apply enhancement
    return cv2.detailEnhance(img, sigma_s=10, sigma_r=0.15) 

# function to expand bounding box
def expand_box(x1, y1, x2, y2, w, h, scale):
    # compute box width and height  
    bw, bh = x2 - x1, y2 - y1            
    
    # compute expansion
    dx, dy = int(bw * scale), int(bh * scale)
    
    # return expanded and clipped box  
    return (                              
        max(0, x1 - dx),
        max(0, y1 - dy),
        min(w, x2 + dx),
        min(h, y2 + dy),
    )
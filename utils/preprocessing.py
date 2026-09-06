"""
Image preprocessing utilities for invoice cleaning.
"""

import cv2
import numpy as np


def deskew_image(image):
    """
    Detect and correct skew angle of text in the image.
    
    Args:
        image: Input image (BGR format)
        
    Returns:
        Deskewed image
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.bitwise_not(gray)
    
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    
    coords = np.column_stack(np.where(thresh > 0))
    
    if len(coords) == 0:
        return image
    
    angle = cv2.minAreaRect(coords)[-1]
    
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(
        image, M, (w, h), 
        flags=cv2.INTER_CUBIC, 
        borderMode=cv2.BORDER_REPLICATE
    )
    
    return rotated


def remove_shadows(image):
    """
    Remove uneven lighting and shadows using background estimation.
    
    Args:
        image: Input image (BGR format)
        
    Returns:
        Shadow-corrected image
    """
    rgb_planes = cv2.split(image)
    result_planes = []
    
    for plane in rgb_planes:
        dilated_img = cv2.dilate(plane, np.ones((7, 7), np.uint8))
        bg_img = cv2.medianBlur(dilated_img, 21)
        diff_img = 255 - cv2.absdiff(plane, bg_img)
        norm_img = cv2.normalize(
            diff_img, None, 
            alpha=0, beta=255, 
            norm_type=cv2.NORM_MINMAX, 
            dtype=cv2.CV_8UC1
        )
        result_planes.append(norm_img)
    
    return cv2.merge(result_planes)


def enhance_contrast(image):
    """
    Apply CLAHE (Contrast Limited Adaptive Histogram Equalization).
    
    Args:
        image: Input image (BGR format)
        
    Returns:
        Contrast-enhanced image
    """
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    
    limg = cv2.merge((cl, a, b))
    final = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    
    return final


def preprocess_pipeline(image_path, output_path="temp_processed.jpg"):
    """
    Run complete preprocessing pipeline on an image.
    
    Args:
        image_path: Path to input image
        output_path: Path to save processed image
        
    Returns:
        Path to processed image
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not load image at {image_path}")
    
    # Deskew
    try:
        img = deskew_image(img)
    except Exception as e:
        print(f"⚠️  Deskew failed ({e}), skipping step.")
    
    # Shadow removal
    img = remove_shadows(img)
    
    # Contrast enhancement
    img = enhance_contrast(img)
    
    # Save processed image
    cv2.imwrite(output_path, img)
    
    return output_path
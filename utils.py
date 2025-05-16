import os
import numpy as np
import cv2
import pydicom
from skimage import exposure
from skimage.restoration import denoise_bilateral

def load_image(path):
    ext = os.path.splitext(path)[1].lower()
    
    if ext in ['.jpg', '.jpeg', '.png']:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        return img
    elif ext == '.dcm':
        ds = pydicom.dcmread(path)
        img = ds.pixel_array.astype(np.float32)
        return img, ds
    else:
        raise ValueError(f"Unsupported file format: {ext}")

def preprocess_image(img):
    # Normalize (z-score)
    img = img.astype(np.float32)
    mean, std = np.mean(img), np.std(img)
    if std != 0:
        img = (img - mean) / std
    
    # Histogram Equalization
    img = exposure.equalize_hist(img)
    
    # Denoising
    img = denoise_bilateral(img, sigma_color=0.05, sigma_spatial=15, multichannel=False)

    # Rescale to 0-255 and convert to uint8
    img = (img * 255).astype(np.uint8)
    
    return img

def extract_dicom_metadata(ds):
    metadata = {
        "PatientID": ds.get("PatientID", "Unknown"),
        "Modality": ds.get("Modality", "Unknown"),
        "StudyDate": ds.get("StudyDate", "Unknown"),
        "Manufacturer": ds.get("Manufacturer", "Unknown"),
        "SliceThickness": ds.get("SliceThickness", "Unknown"),
        "PixelSpacing": ds.get("PixelSpacing", "Unknown"),
    }
    return metadata

def preprocess_and_save(input_path, output_path):
    if input_path.lower().endswith('.dcm'):
        img, ds = load_image(input_path)
        metadata = extract_dicom_metadata(ds)
    else:
        img = load_image(input_path)
        metadata = {}

    preprocessed_img = preprocess_image(img)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, preprocessed_img)

    return preprocessed_img, metadata

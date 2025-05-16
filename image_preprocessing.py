class CTMRIDatasetProcessor:
    """
    Class for handling CT-to-MRI cGAN dataset processing including:
    - Extracting data from the zip file
    - Loading and organizing paired CT and MRI images
    - Preparing data for model input
    """
    
    def __init__(self, zip_path: str, extract_dir: str = None):
        """
        Initialize the dataset processor
        
        Args:
            zip_path: Path to the downloaded zip file
            extract_dir: Directory to extract files to (default: same directory as zip)
        """
        self.zip_path = zip_path
        
        if extract_dir is None:
            self.extract_dir = os.path.dirname(zip_path)
        else:
            self.extract_dir = extract_dir
            
        self.dataset_dir = os.path.join(self.extract_dir, 'ct-to-mri-cgan')
        self.preprocessor = ImagePreprocessor()
        
    def extract_dataset(self, force_extract: bool = False) -> None:
        """
        Extract the dataset from the zip file
        
        Args:
            force_extract: Whether to force extraction even if directory exists
        """
        if os.path.exists(self.dataset_dir) and not force_extract:
            logger.info(f"Dataset directory already exists at {self.dataset_dir}")
            return
            
        logger.info(f"Extracting dataset from {self.zip_path} to {self.extract_dir}")
        
        try:
            with zipfile.ZipFile(self.zip_path, 'r') as zip_ref:
                zip_ref.extractall(self.extract_dir)
            logger.info("Dataset extraction complete")
        except Exception as e:
            logger.error(f"Error extracting dataset: {e}")
            raise
    
    def get_dataset_structure(self) -> Dict[str, Any]:
        """
        Analyze and return the structure of the extracted dataset
        
        Returns:
            Dictionary describing the dataset structure
        """
        structure = {
            "root_dir": self.dataset_dir,
            "directories": [],
            "files": [],
            "ct_files": [],
            "mri_files": [],
            "other_files": []
        }
        
        if not os.path.exists(self.dataset_dir):
            logger.error(f"Dataset directory does not exist: {self.dataset_dir}")
            return structure
            
        # Walk through the directory structure
        for root, dirs, files in os.walk(self.dataset_dir):
            rel_path = os.path.relpath(root, self.dataset_dir)
            if rel_path == '.':
                rel_path = ''
                
            dir_info = {
                "path": rel_path,
                "file_count": len(files)
            }
            structure["directories"].append(dir_info)
            
            # Categorize files
            for file in files:
                file_path = os.path.join(root, file)
                rel_file_path = os.path.join(rel_path, file)
                
                file_info = {
                    "name": file,
                    "path": rel_file_path,
                    "full_path": file_path,
                    "size": os.path.getsize(file_path)
                }
                
                structure["files"].append(file_info)
                
                # Categorize by file type
                lower_name = file.lower()
                if 'ct' in lower_name:
                    structure["ct_files"].append(file_info)
                elif 'mri' in lower_name or 'mr' in lower_name:
                    structure["mri_files"].append(file_info)
                else:
                    structure["other_files"].append(file_info)
        
        return structure
    
    def load_sample_image_pair(self) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Load a sample CT-MRI image pair from the dataset
        
        Returns:
            Tuple containing:
                - CT image as numpy array
                - MRI image as numpy array
                - Dictionary of metadata
        """
        structure = self.get_dataset_structure()
        
        # Find paired images (this will depend on the actual dataset structure)
        # This is a placeholder implementation - adjust based on actual dataset organization
        ct_files = structure["ct_files"]
        mri_files = structure["mri_files"]
        
        if not ct_files or not mri_files:
            raise ValueError("Could not find CT and MRI files in the dataset")
            
        # For demonstration, we'll just use the first CT and MRI files
        ct_file_path = ct_files[0]["full_path"]
        mri_file_path = mri_files[0]["full_path"]
        
        # Load images
        ct_img, ct_metadata = self.preprocessor.load_image(ct_file_path)
        mri_img, mri_metadata = self.preprocessor.load_image(mri_file_path)
        
        # Combine metadata
        combined_metadata = {
            "ct": ct_metadata,
            "mri": mri_metadata,
            "ct_path": ct_file_path,
            "mri_path": mri_file_path
        }
        
        return ct_img, mri_img, combined_metadata
    
    def preprocess_image_pair(self, ct_img: np.ndarray, mri_img: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply appropriate preprocessing to a CT-MRI image pair
        
        Args:
            ct_img: CT image as numpy array
            mri_img: MRI image as numpy array
            
        Returns:
            Tuple containing preprocessed CT and MRI images
        """
        # Apply modality-specific preprocessing
        processed_ct = self.preprocessor.preprocess_image(
            ct_img, 
            apply_hist_equal=True,
            apply_gaussian=True,
            gaussian_sigma=0.5,
            apply_norm=True
        )
        
        processed_mri = self.preprocessor.preprocess_image(
            mri_img,
            apply_hist_equal=True,
            apply_gaussian=True,
            gaussian_sigma=0.8,  # MRI might need different smoothing
            apply_norm=True
        )
        
        return processed_ct, processed_mri
    
    def visualize_image_pair(self, ct_img: np.ndarray, mri_img: np.ndarray, 
                            processed_ct: np.ndarray = None, processed_mri: np.ndarray = None,
                            save_path: str = None) -> None:
        """
        Visualize original and processed CT-MRI image pairs
        
        Args:
            ct_img: Original CT image
            mri_img: Original MRI image
            processed_ct: Processed CT image (optional)
            processed_mri: Processed MRI image (optional)
            save_path: Path to save visualization (optional)
        """
        if processed_ct is None and processed_mri is None:
            fig, axes = plt.subplots(1, 2, figsize=(10, 5))
            axes[0].imshow(ct_img, cmap='gray')
            axes[0].set_title('CT Image')
            axes[0].axis('off')
            
            axes[1].imshow(mri_img, cmap='gray')
            axes[1].set_title('MRI Image')
            axes[1].axis('off')
        else:
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            
            axes[0, 0].imshow(ct_img, cmap='gray')
            axes[0, 0].set_title('Original CT')
            axes[0, 0].axis('off')
            
            axes[0, 1].imshow(mri_img, cmap='gray')
            axes[0, 1].set_title('Original MRI')
            axes[0, 1].axis('off')
            
            if processed_ct is not None:
                axes[1, 0].imshow(processed_ct, cmap='gray')
                axes[1, 0].set_title('Processed CT')
                axes[1, 0].axis('off')
                
            if processed_mri is not None:
                axes[1, 1].imshow(processed_mri, cmap='gray')
                axes[1, 1].set_title('Processed MRI')
                axes[1, 1].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Visualization saved to {save_path}")
            
        plt.show()import os
import numpy as np
import cv2
import pydicom
from skimage import exposure, transform, filters, util
from skimage.filters import gaussian
import nibabel as nib
import zipfile
import logging
import matplotlib.pyplot as plt
from PIL import Image
import SimpleITK as sitk
from tqdm import tqdm
import pandas as pd
from scipy import ndimage
import glob
import json
from typing import Dict, Tuple, Any, Optional, Union, List

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ImagePreprocessor:
    """
    Class for handling medical image preprocessing tasks including:
    - Loading DICOM, PNG, JPEG images, and NIfTI files
    - Applying preprocessing techniques (histogram equalization, noise reduction, normalization)
    - Extracting metadata from DICOM files
    """
    
    def __init__(self):
        self.supported_extensions = ['.dcm', '.png', '.jpg', '.jpeg', '.nii', '.nii.gz']
        
    def load_image(self, file_path: str) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Load a medical image from file path and extract metadata if available
        
        Args:
            file_path: Path to the image file
            
        Returns:
            Tuple containing:
                - Image as numpy array
                - Dictionary of metadata (empty for non-DICOM/NIfTI images)
        """
        file_ext = os.path.splitext(file_path)[1].lower()
        if file_path.lower().endswith('.nii.gz'):
            file_ext = '.nii.gz'
        
        if file_ext not in self.supported_extensions:
            raise ValueError(f"Unsupported file extension: {file_ext}. Supported: {self.supported_extensions}")
        
        if file_ext == '.dcm':
            return self._load_dicom(file_path)
        elif file_ext in ['.nii', '.nii.gz']:
            return self._load_nifti(file_path)
        else:
            return self._load_regular_image(file_path)
    
    def _load_dicom(self, file_path: str) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Load a DICOM image and extract its metadata"""
        try:
            dicom_data = pydicom.dcmread(file_path)
            
            # Extract image data
            img = dicom_data.pixel_array
            
            # Convert to float and normalize if needed
            if img.dtype != np.float32:
                img = img.astype(np.float32)
            
            # Normalize based on modality window settings
            if hasattr(dicom_data, 'WindowCenter') and hasattr(dicom_data, 'WindowWidth'):
                center = dicom_data.WindowCenter
                width = dicom_data.WindowWidth
                
                # Handle when these attributes are sequences
                if isinstance(center, pydicom.multival.MultiValue):
                    center = center[0]
                if isinstance(width, pydicom.multival.MultiValue):
                    width = width[0]
                    
                img_min = center - width // 2
                img_max = center + width // 2
                img = np.clip(img, img_min, img_max)
                img = (img - img_min) / (img_max - img_min)
            else:
                # Simple min-max normalization if window settings not available
                if img.max() > 0:
                    img = img / img.max()
            
            # Extract metadata
            metadata = self._extract_dicom_metadata(dicom_data)
            
            return img, metadata
            
        except Exception as e:
            logger.error(f"Error loading DICOM image: {e}")
            raise
    
    def _load_nifti(self, file_path: str) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Load a NIfTI image and extract its metadata"""
        try:
            nifti_img = nib.load(file_path)
            
            # Get the data as a numpy array
            img_data = nifti_img.get_fdata()
            
            # If it's a 3D volume, take a middle slice for 2D processing
            if len(img_data.shape) == 3:
                middle_slice_idx = img_data.shape[2] // 2
                img = img_data[:, :, middle_slice_idx]
            else:
                img = img_data
            
            # Normalize the image
            if img.max() > 0:
                img = img / img.max()
            
            # Extract metadata
            metadata = {
                'affine': nifti_img.affine.tolist(),
                'dimensions': img_data.shape,
                'spacing': nifti_img.header.get_zooms(),
                'datatype': str(nifti_img.header.get_data_dtype()),
                'file_path': file_path
            }
            
            return img.astype(np.float32), metadata
            
        except Exception as e:
            logger.error(f"Error loading NIfTI image: {e}")
            raise
    
    def _load_regular_image(self, file_path: str) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Load a regular image (PNG, JPEG)"""
        try:
            # Using PIL for better handling of various image formats
            with Image.open(file_path) as pil_img:
                # Convert to grayscale if it's a color image
                if pil_img.mode != 'L':
                    pil_img = pil_img.convert('L')
                
                # Convert to numpy array
                img = np.array(pil_img).astype(np.float32)
            
            # Normalize to 0-1 range
            img = img / 255.0
            
            # Extract basic metadata
            metadata = {
                'width': img.shape[1],
                'height': img.shape[0],
                'file_path': file_path,
                'format': os.path.splitext(file_path)[1][1:].upper()
            }
            
            return img, metadata
            
        except Exception as e:
            logger.error(f"Error loading image: {e}")
            raise
    
    def _extract_dicom_metadata(self, dicom_data: pydicom.dataset.FileDataset) -> Dict[str, Any]:
        """
        Extract relevant metadata from DICOM file
        
        Args:
            dicom_data: pydicom dataset
            
        Returns:
            Dictionary containing metadata fields
        """
        metadata = {}
        
        # Define fields to extract with safe fallbacks
        fields = {
            'PatientID': 'Anonymous',
            'PatientName': 'Anonymous',
            'PatientBirthDate': None,
            'PatientSex': None,
            'Modality': None,
            'StudyDescription': None,
            'SeriesDescription': None,
            'StudyDate': None,
            'StudyTime': None,
            'PixelSpacing': None,
            'SliceThickness': None,
            'Manufacturer': None,
            'InstitutionName': None
        }
        
        # Extract available fields
        for field, default in fields.items():
            if hasattr(dicom_data, field):
                metadata[field] = str(getattr(dicom_data, field))
            else:
                if default is not None:
                    metadata[field] = default
        
        # Add DICOM-specific metadata
        metadata['ImageSize'] = dicom_data.pixel_array.shape
        
        # Anonymize patient identifiable information for HIPAA compliance
        metadata = self._anonymize_metadata(metadata)
        
        return metadata
    
    def _anonymize_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Anonymize identifiable patient information for HIPAA compliance
        
        Args:
            metadata: Dictionary containing metadata
            
        Returns:
            Anonymized metadata dictionary
        """
        # Fields to anonymize
        identifiable_fields = ['PatientID', 'PatientName', 'PatientBirthDate']
        
        for field in identifiable_fields:
            if field in metadata:
                if field == 'PatientID':
                    # Replace with a hash or anonymized ID
                    # In production, use a consistent anonymization method
                    metadata[field] = f"ANON_{hash(metadata[field]) % 10000:04d}"
                else:
                    # Completely remove other PII
                    metadata[field] = "ANONYMIZED"
                    
        return metadata
    
    def preprocess_image(self, image: np.ndarray, 
                         apply_hist_equal: bool = True,
                         apply_gaussian: bool = True,
                         gaussian_sigma: float = 0.5,
                         apply_norm: bool = True,
                         modality: str = None) -> np.ndarray:
        """
        Apply preprocessing techniques to the image
        
        Args:
            image: Input image as numpy array
            apply_hist_equal: Whether to apply histogram equalization
            apply_gaussian: Whether to apply Gaussian filtering
            gaussian_sigma: Sigma value for Gaussian filter
            apply_norm: Whether to apply z-score normalization
            modality: Image modality ('CT' or 'MRI') for specialized processing
            
        Returns:
            Preprocessed image
        """
        processed_img = image.copy()
        
        # Modality-specific preprocessing adjustments
        if modality == 'CT':
            # CT-specific processing (e.g., window adjustment for bone/soft tissue)
            # Example: Enhance contrast for CT
            processed_img = exposure.rescale_intensity(processed_img)
        elif modality == 'MRI':
            # MRI-specific processing (e.g., bias field correction would go here)
            # For now, just use standard processing
            pass
        
        # Histogram equalization
        if apply_hist_equal:
            # Use CLAHE for more natural-looking results
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            if processed_img.dtype != np.uint8:
                # Convert to uint8 for CLAHE
                processed_img_uint8 = (processed_img * 255).astype(np.uint8)
                processed_img = clahe.apply(processed_img_uint8).astype(np.float32) / 255.0
            else:
                processed_img = clahe.apply(processed_img).astype(np.float32) / 255.0
        
        # Gaussian filtering for noise reduction
        if apply_gaussian:
            processed_img = gaussian(processed_img, sigma=gaussian_sigma)
        
        # Z-score normalization
        if apply_norm:
            mean = np.mean(processed_img)
            std = np.std(processed_img)
            if std > 0:  # Avoid division by zero
                processed_img = (processed_img - mean) / std
            
        return processed_img
    
    def save_preprocessed_image(self, image: np.ndarray, file_path: str) -> None:
        """
        Save the preprocessed image to disk
        
        Args:
            image: Image as numpy array
            file_path: Path to save the image
        """
        try:
            # Ensure the directory exists
            os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
            
            # Normalize to 0-255 for saving
            img_to_save = (image - image.min()) / (image.max() - image.min() + 1e-8) * 255
            img_to_save = img_to_save.astype(np.uint8)
            
            cv2.imwrite(file_path, img_to_save)
            logger.info(f"Saved preprocessed image to {file_path}")
            
        except Exception as e:
            logger.error(f"Error saving image: {e}")
            raise
            
    def extract_slices_from_volume(self, volume_data: np.ndarray, 
                                 axis: int = 2, 
                                 slice_indices: List[int] = None) -> List[np.ndarray]:
        """
        Extract 2D slices from a 3D volume
        
        Args:
            volume_data: 3D volume as numpy array
            axis: Axis along which to extract slices (0=sagittal, 1=coronal, 2=axial)
            slice_indices: List of indices to extract (None = all slices)
            
        Returns:
            List of 2D slices as numpy arrays
        """
        if len(volume_data.shape) != 3:
            raise ValueError(f"Expected 3D volume, got shape {volume_data.shape}")
            
        if slice_indices is None:
            # Extract all slices
            num_slices = volume_data.shape[axis]
            slice_indices = range(num_slices)
            
        slices = []
        for idx in slice_indices:
            if axis == 0:
                slice_data = volume_data[idx, :, :]
            elif axis == 1:
                slice_data = volume_data[:, idx, :]
            else:  # axis == 2
                slice_data = volume_data[:, :, idx]
                
            slices.append(slice_data)
            
        return slices

# Usage example
if __name__ == "__main__":
    preprocessor = ImagePreprocessor()
    
    # Example with a DICOM file
    # image, metadata = preprocessor.load_image("path/to/image.dcm")
    # processed_image = preprocessor.preprocess_image(image)
    # preprocessor.save_preprocessed_image(processed_image, "path/to/output.png")
    
    print("Image Preprocessor module")
"""
Text detection and reconstruction module for signboards and text in images.
"""

import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional
import logging
import re

# Try to import OCR libraries, make them optional
try:
    import pytesseract
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False

try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TextReconstruction:
    """Detect and reconstruct blurred/faded text on signboards."""
    
    def __init__(self):
        # Initialize EasyOCR reader (supports multiple languages)
        self.easyocr_reader = None
        if EASYOCR_AVAILABLE:
            try:
                self.easyocr_reader = easyocr.Reader(['en'], gpu=False)
            except Exception as e:
                logger.warning(f"EasyOCR initialization failed: {e}. Using Tesseract only.")
                self.easyocr_reader = None
        else:
            logger.warning("EasyOCR not available. Text reconstruction will be limited.")
            
    def detect_text_regions(self, image: np.ndarray) -> List[Dict]:
        """
        Detect regions containing text.
        
        Args:
            image: Input image
            
        Returns:
            List of text regions with bounding boxes
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        # Apply adaptive thresholding to find text-like regions
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
        )
        
        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        text_regions = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            
            # Filter by aspect ratio and size (text-like regions)
            aspect_ratio = w / h if h > 0 else 0
            area = w * h
            
            # Typical text characteristics
            if 0.2 < aspect_ratio < 10 and area > 100:
                text_regions.append({
                    'bbox': (x, y, w, h),
                    'region': image[y:y+h, x:x+w] if len(image.shape) == 3 else gray[y:y+h, x:x+w]
                })
                
        return text_regions
    
    def enhance_text_region(self, text_region: np.ndarray) -> np.ndarray:
        """
        Enhance a text region for better OCR.
        
        Args:
            text_region: Text region image
            
        Returns:
            Enhanced text region
        """
        if text_region.size == 0:
            return text_region
            
        # Convert to grayscale if needed
        if len(text_region.shape) == 3:
            gray = cv2.cvtColor(text_region, cv2.COLOR_BGR2GRAY)
        else:
            gray = text_region.copy()
            
        # Resize if too small (improves OCR accuracy)
        if gray.shape[0] < 30 or gray.shape[1] < 30:
            scale = max(30 / gray.shape[0], 30 / gray.shape[1])
            new_h = int(gray.shape[0] * scale)
            new_w = int(gray.shape[1] * scale)
            gray = cv2.resize(gray, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        
        # Apply sharpening
        kernel = np.array([[-1, -1, -1],
                          [-1,  9, -1],
                          [-1, -1, -1]])
        sharpened = cv2.filter2D(gray, -1, kernel)
        
        # Apply denoising
        denoised = cv2.fastNlMeansDenoising(sharpened, None, h=10)
        
        # Enhance contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(denoised)
        
        # Apply morphological operations to clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        enhanced = cv2.morphologyEx(enhanced, cv2.MORPH_CLOSE, kernel)
        
        return enhanced
    
    def reconstruct_text(self, image: np.ndarray) -> Tuple[List[Dict], np.ndarray]:
        """
        Detect and reconstruct text in the image.
        
        Args:
            image: Input image
            
        Returns:
            List of detected text with locations and reconstructed text,
            Image with text regions highlighted
        """
        text_results = []
        result_image = image.copy()
        
        # Detect text regions
        text_regions = self.detect_text_regions(image)
        
        # Process each region
        for region_info in text_regions:
            x, y, w, h = region_info['bbox']
            region = region_info['region']
            
            # Enhance the region
            enhanced_region = self.enhance_text_region(region)
            
            # Try OCR with both methods
            detected_text = self._ocr_text(enhanced_region)
            
            if detected_text and len(detected_text.strip()) > 0:
                text_results.append({
                    'text': detected_text,
                    'bbox': (x, y, w, h),
                    'confidence': 0.8  # Placeholder
                })
                
                # Draw bounding box on result image
                cv2.rectangle(result_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                
                # Add text label
                cv2.putText(result_image, detected_text[:30], (x, y - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        return text_results, result_image
    
    def _ocr_text(self, text_region: np.ndarray) -> Optional[str]:
        """
        Perform OCR on a text region using multiple methods.
        
        Args:
            text_region: Enhanced text region
            
        Returns:
            Detected text or None
        """
        # Try EasyOCR first (better for blurred text)
        if self.easyocr_reader is not None:
            try:
                results = self.easyocr_reader.readtext(text_region)
                if results:
                    # Combine all detected text
                    text = ' '.join([result[1] for result in results])
                    return text.strip()
            except Exception as e:
                logger.debug(f"EasyOCR failed: {e}")
        
        # Fallback to Tesseract
        if TESSERACT_AVAILABLE:
            try:
                # Configure Tesseract for better results
                custom_config = r'--oem 3 --psm 6'
                text = pytesseract.image_to_string(text_region, config=custom_config)
                text = re.sub(r'\s+', ' ', text).strip()
                return text if text else None
            except Exception as e:
                logger.debug(f"Tesseract failed: {e}")
                return None
        else:
            logger.debug("Tesseract not available. Cannot perform OCR.")
            return None
    
    def restore_blurred_text(self, 
                            original_image: np.ndarray,
                            enhanced_image: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """
        Restore blurred text on signboards.
        
        Args:
            original_image: Original image
            enhanced_image: Enhanced image
            
        Returns:
            Image with restored text and report
        """
        report = {
            'text_regions_found': 0,
            'text_detected': [],
            'operations': []
        }
        
        # Detect and reconstruct text
        text_results, result_image = self.reconstruct_text(enhanced_image)
        
        report['text_regions_found'] = len(text_results)
        report['text_detected'] = [
            {
                'text': result['text'],
                'location': result['bbox']
            }
            for result in text_results
        ]
        report['operations'].append('text_reconstruction')
        
        return result_image, report


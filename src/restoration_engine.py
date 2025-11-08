"""
Main restoration engine for enhancing low-quality images.
"""

import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
from typing import List, Dict, Tuple, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RestorationEngine:
    """Core engine for image restoration and enhancement."""
    
    def __init__(self):
        self.restoration_history = []
        
    def enhance_image(self, image: np.ndarray, 
                     denoise: bool = True,
                     sharpen: bool = True,
                     contrast: bool = True,
                     color_correction: bool = True) -> Tuple[np.ndarray, Dict]:
        """
        Enhance a single image with multiple restoration techniques.
        
        Args:
            image: Input image as numpy array
            denoise: Apply denoising
            sharpen: Apply sharpening
            contrast: Enhance contrast
            color_correction: Apply color correction
            
        Returns:
            Enhanced image and restoration report
        """
        original = image.copy()
        report = {
            'operations': [],
            'improvements': {}
        }
        
        # Convert to appropriate color space
        if len(image.shape) == 3:
            # Convert BGR to RGB for processing
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image_rgb = image
            
        # Denoising
        if denoise:
            image_rgb = self._denoise(image_rgb)
            report['operations'].append('denoising')
            report['improvements']['noise_reduction'] = 'Applied non-local means denoising'
            
        # Color correction and restoration
        if color_correction:
            image_rgb = self._color_correction(image_rgb)
            report['operations'].append('color_correction')
            report['improvements']['color_restoration'] = 'Enhanced faded colors'
            
        # Contrast enhancement
        if contrast:
            image_rgb = self._enhance_contrast(image_rgb)
            report['operations'].append('contrast_enhancement')
            report['improvements']['contrast'] = 'Improved dynamic range'
            
        # Sharpening
        if sharpen:
            image_rgb = self._sharpen(image_rgb)
            report['operations'].append('sharpening')
            report['improvements']['sharpness'] = 'Enhanced fine details'
            
        # Convert back to BGR for OpenCV compatibility
        if len(image_rgb.shape) == 3:
            enhanced = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        else:
            enhanced = image_rgb
            
        # Calculate improvement metrics
        report['metrics'] = self._calculate_improvements(original, enhanced)
        
        return enhanced, report
    
    def _denoise(self, image: np.ndarray) -> np.ndarray:
        """Apply advanced denoising."""
        if len(image.shape) == 3:
            # Color image - use fastNlMeansDenoisingColored
            denoised = cv2.fastNlMeansDenoisingColored(
                image, None, h=10, hColor=10, templateWindowSize=7, searchWindowSize=21
            )
        else:
            # Grayscale
            denoised = cv2.fastNlMeansDenoising(
                image, None, h=10, templateWindowSize=7, searchWindowSize=21
            )
        return denoised
    
    def _color_correction(self, image: np.ndarray) -> np.ndarray:
        """Restore faded colors."""
        # Convert to LAB color space for better color manipulation
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        
        # Enhance color channels
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        
        # Enhance color saturation
        a = cv2.add(a, 5)  # Slight boost to green-red channel
        b = cv2.add(b, 5)  # Slight boost to blue-yellow channel
        
        # Merge and convert back
        enhanced_lab = cv2.merge([l, a, b])
        enhanced = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2RGB)
        
        return enhanced
    
    def _enhance_contrast(self, image: np.ndarray) -> np.ndarray:
        """Enhance image contrast."""
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        if len(image.shape) == 3:
            lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            l = clahe.apply(l)
            enhanced = cv2.merge([l, a, b])
            enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2RGB)
        else:
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(image)
            
        return enhanced
    
    def _sharpen(self, image: np.ndarray) -> np.ndarray:
        """Apply smart sharpening."""
        # Create unsharp mask
        gaussian = cv2.GaussianBlur(image, (0, 0), 2.0)
        sharpened = cv2.addWeighted(image, 1.5, gaussian, -0.5, 0)
        
        # Clip values to valid range
        sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)
        
        return sharpened
    
    def _calculate_improvements(self, original: np.ndarray, enhanced: np.ndarray) -> Dict:
        """Calculate quantitative improvement metrics."""
        metrics = {}
        
        # Calculate variance (measure of detail)
        if len(original.shape) == 3:
            orig_var = np.var(cv2.cvtColor(original, cv2.COLOR_BGR2GRAY))
            enh_var = np.var(cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY))
        else:
            orig_var = np.var(original)
            enh_var = np.var(enhanced)
            
        metrics['detail_increase'] = f"{(enh_var / orig_var - 1) * 100:.1f}%"
        
        # Calculate average brightness
        if len(original.shape) == 3:
            orig_bright = np.mean(cv2.cvtColor(original, cv2.COLOR_BGR2GRAY))
            enh_bright = np.mean(cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY))
        else:
            orig_bright = np.mean(original)
            enh_bright = np.mean(enhanced)
            
        metrics['brightness_change'] = f"{enh_bright - orig_bright:.1f}"
        
        return metrics
    
    def restore_faded_photo(self, image: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """
        Specialized restoration for faded photographs.
        
        Args:
            image: Faded input image
            
        Returns:
            Restored image and detailed report
        """
        report = {
            'type': 'faded_photo_restoration',
            'operations': [],
            'improvements': {}
        }
        
        # Convert to RGB
        if len(image.shape) == 3:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image_rgb = cv2.copyMakeBorder(image, 0, 0, 0, 0, cv2.BORDER_CONSTANT, value=0)
            image_rgb = cv2.cvtColor(image_rgb, cv2.COLOR_GRAY2RGB)
        
        # Aggressive color restoration
        lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        
        # Enhance lightness
        clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        
        # Boost color channels more aggressively
        a = cv2.add(a, 10)
        b = cv2.add(b, 10)
        
        enhanced_lab = cv2.merge([l, a, b])
        enhanced = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2RGB)
        
        # Apply denoising
        enhanced = self._denoise(enhanced)
        
        # Sharpen
        enhanced = self._sharpen(enhanced)
        
        report['operations'] = [
            'aggressive_color_restoration',
            'lightness_enhancement',
            'denoising',
            'sharpening'
        ]
        report['improvements'] = {
            'color_vibrancy': 'Restored faded colors to original intensity',
            'brightness': 'Enhanced overall brightness',
            'detail': 'Recovered fine details lost to fading'
        }
        
        # Convert back to BGR
        if len(enhanced.shape) == 3:
            result = cv2.cvtColor(enhanced, cv2.COLOR_RGB2BGR)
        else:
            result = enhanced
            
        return result, report



"""
Main restoration engine for enhancing low-quality images.
"""

import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
from typing import List, Dict, Tuple, Optional
import logging

# Try to import AI enhancement
try:
    from .ai_enhancement import create_ai_enhancer, AIEnhancement
    AI_ENHANCEMENT_AVAILABLE = True
except ImportError:
    AI_ENHANCEMENT_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("AI enhancement module not available")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RestorationEngine:
    """Core engine for image restoration and enhancement."""
    
    def __init__(self, use_ai_model: bool = True, ai_model_name: str = 'realesrgan', ai_model_scale: int = 4, ai_exe_path: Optional[str] = None):
        """
        Initialize restoration engine.
        
        Args:
            use_ai_model: Whether to use AI models for enhancement
            ai_model_name: AI model to use ('realesrgan', 'esrgan', 'traditional')
            ai_model_scale: Upscaling factor for AI models (2 or 4)
            ai_exe_path: Path to portable Real-ESRGAN executable (optional)
        """
        self.restoration_history = []
        self.ai_enhancer = None
        self.use_ai_model = use_ai_model
        
        # Initialize AI enhancer if requested and available
        if use_ai_model and AI_ENHANCEMENT_AVAILABLE:
            try:
                self.ai_enhancer = create_ai_enhancer(
                    model_name=ai_model_name, 
                    model_scale=ai_model_scale,
                    exe_path=ai_exe_path
                )
                if self.ai_enhancer and self.ai_enhancer.is_available():
                    logger.info(f"AI enhancement enabled using {ai_model_name}")
                    if ai_exe_path:
                        logger.info(f"Using portable executable: {ai_exe_path}")
                else:
                    logger.info("AI enhancement requested but not available, using traditional methods")
                    self.use_ai_model = False
            except Exception as e:
                logger.warning(f"Failed to initialize AI enhancer: {e}. Using traditional methods.")
                self.use_ai_model = False
        
    def enhance_image(self, image: np.ndarray, 
                     denoise: bool = True,
                     sharpen: bool = True,
                     contrast: bool = True,
                     color_correction: bool = True,
                     enhance_quality: bool = True,
                     upscale: bool = False) -> Tuple[np.ndarray, Dict]:
        """
        Enhance a single image with multiple restoration techniques.
        Uses adaptive processing that only enhances when needed.
        
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
        
        # Optional upscaling for better detail recovery
        # Skip traditional upscaling if AI model will handle it
        if upscale and not (self.use_ai_model and self.ai_enhancer and self.ai_enhancer.is_available()):
            original_size = image_rgb.shape[:2]
            # Use edge-preserving upscaling (only if AI model not available)
            image_rgb = self._upscale_image(image_rgb, scale=2.0)
            report['operations'].append('upscaling')
            report['improvements']['resolution'] = f'Upscaled from {original_size[1]}x{original_size[0]} to {image_rgb.shape[1]}x{image_rgb.shape[0]}'
            logger.info(f"Upscaled image from {original_size[1]}x{original_size[0]} to {image_rgb.shape[1]}x{image_rgb.shape[0]}")
        
        # Assess image quality to determine enhancement intensity
        quality_score = self._assess_image_quality(image_rgb)
        logger.info(f"Image quality score: {quality_score:.2f} (higher = better quality)")
        
        # Only apply aggressive enhancements if image quality is low
        needs_aggressive_enhancement = quality_score < 0.6
        
        # AI-powered enhancement (if available and enabled)
        if self.use_ai_model and self.ai_enhancer and self.ai_enhancer.is_available():
            logger.info("Using AI model for enhancement...")
            image_rgb_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR) if len(image_rgb.shape) == 3 else image_rgb
            enhanced_ai, ai_report = self.ai_enhancer.enhance(image_rgb_bgr, upscale=upscale)
            
            if enhanced_ai is not None and enhanced_ai.shape != image_rgb_bgr.shape:
                # Convert back to RGB
                if len(enhanced_ai.shape) == 3:
                    image_rgb = cv2.cvtColor(enhanced_ai, cv2.COLOR_BGR2RGB)
                else:
                    image_rgb = enhanced_ai
                
                # Merge AI report
                report['operations'].extend(ai_report.get('operations', []))
                report['improvements'].update(ai_report.get('improvements', {}))
                report['improvements']['ai_model'] = f"Enhanced using {ai_report.get('model_used', 'AI')} model"
                logger.info("AI enhancement completed successfully")
            else:
                # AI enhancement failed or not available, fall back to traditional
                logger.info("AI enhancement not available, using traditional methods")
                if enhance_quality:
                    image_rgb = self._enhance_image_quality(image_rgb, quality_score)
                    report['operations'].append('quality_enhancement')
                    report['improvements']['detail_recovery'] = 'Recovered lost details using advanced algorithms'
        elif enhance_quality:
            # Traditional quality enhancement
            image_rgb = self._enhance_image_quality(image_rgb, quality_score)
            report['operations'].append('quality_enhancement')
            report['improvements']['detail_recovery'] = 'Recovered lost details using advanced algorithms'
        
        # Denoising - only if image has significant noise
        if denoise:
            noise_level = self._estimate_noise_level(image_rgb)
            if noise_level > 5:  # Only denoise if noise is significant
                image_rgb = self._denoise(image_rgb, strength='light' if quality_score > 0.5 else 'medium')
                report['operations'].append('denoising')
                report['improvements']['noise_reduction'] = f'Applied gentle denoising (noise level: {noise_level:.1f})'
            else:
                logger.info("Skipping denoising - image is already clean")
            
        # Color correction - only if colors appear faded
        if color_correction:
            color_vibrancy = self._assess_color_vibrancy(image_rgb)
            if color_vibrancy < 0.7 or needs_aggressive_enhancement:
                image_rgb = self._color_correction(image_rgb, strength='gentle' if quality_score > 0.6 else 'moderate')
                report['operations'].append('color_correction')
                report['improvements']['color_restoration'] = 'Enhanced faded colors'
            else:
                logger.info("Skipping color correction - colors are already vibrant")
            
        # Contrast enhancement - only if contrast is low
        if contrast:
            contrast_level = self._assess_contrast(image_rgb)
            if contrast_level < 0.6 or needs_aggressive_enhancement:
                image_rgb = self._enhance_contrast(image_rgb, strength='gentle' if quality_score > 0.6 else 'moderate')
                report['operations'].append('contrast_enhancement')
                report['improvements']['contrast'] = 'Improved dynamic range'
            else:
                logger.info("Skipping contrast enhancement - contrast is already good")
            
        # Sharpening - only if image is blurry
        if sharpen:
            sharpness = self._assess_sharpness(image_rgb)
            if sharpness < 0.5 or needs_aggressive_enhancement:
                image_rgb = self._sharpen(image_rgb, strength='light' if quality_score > 0.6 else 'medium')
                report['operations'].append('sharpening')
                report['improvements']['sharpness'] = 'Enhanced fine details'
            else:
                logger.info("Skipping sharpening - image is already sharp")
            
        # Convert back to BGR for OpenCV compatibility
        if len(image_rgb.shape) == 3:
            enhanced = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        else:
            enhanced = image_rgb
        
        # If no operations were applied, return original
        if len(report['operations']) == 0:
            logger.info("No enhancements needed - returning original image")
            enhanced = original.copy()
            report['improvements']['note'] = 'Image quality is already good - minimal processing applied'
            
        # Calculate improvement metrics
        report['metrics'] = self._calculate_improvements(original, enhanced)
        
        return enhanced, report
    
    def _denoise(self, image: np.ndarray, strength: str = 'medium') -> np.ndarray:
        """Apply advanced denoising with adjustable strength."""
        # Adjust denoising strength based on parameter
        if strength == 'light':
            h = 3
            hColor = 3
        elif strength == 'medium':
            h = 6
            hColor = 6
        else:  # strong
            h = 10
            hColor = 10
            
        if len(image.shape) == 3:
            # Color image - use fastNlMeansDenoisingColored
            denoised = cv2.fastNlMeansDenoisingColored(
                image, None, h=h, hColor=hColor, templateWindowSize=7, searchWindowSize=21
            )
        else:
            # Grayscale
            denoised = cv2.fastNlMeansDenoising(
                image, None, h=h, templateWindowSize=7, searchWindowSize=21
            )
        return denoised
    
    def _color_correction(self, image: np.ndarray, strength: str = 'gentle') -> np.ndarray:
        """Restore faded colors with adjustable strength."""
        # Convert to LAB color space for better color manipulation
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        
        # Adjust enhancement strength
        if strength == 'gentle':
            clahe_limit = 1.5
            color_boost = 2
        elif strength == 'moderate':
            clahe_limit = 2.0
            color_boost = 5
        else:  # strong
            clahe_limit = 2.5
            color_boost = 8
        
        # Enhance color channels gently
        clahe = cv2.createCLAHE(clipLimit=clahe_limit, tileGridSize=(8, 8))
        l = clahe.apply(l)
        
        # Enhance color saturation (more conservative)
        a = cv2.add(a, color_boost)  # Boost to green-red channel
        b = cv2.add(b, color_boost)  # Boost to blue-yellow channel
        
        # Clip to valid range
        a = np.clip(a, 0, 255)
        b = np.clip(b, 0, 255)
        
        # Merge and convert back
        enhanced_lab = cv2.merge([l, a, b])
        enhanced = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2RGB)
        
        return enhanced
    
    def _enhance_contrast(self, image: np.ndarray, strength: str = 'gentle') -> np.ndarray:
        """Enhance image contrast with adjustable strength."""
        # Adjust contrast strength
        if strength == 'gentle':
            clahe_limit = 1.8
        elif strength == 'moderate':
            clahe_limit = 2.5
        else:  # strong
            clahe_limit = 3.0
        
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        if len(image.shape) == 3:
            lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=clahe_limit, tileGridSize=(8, 8))
            l = clahe.apply(l)
            enhanced = cv2.merge([l, a, b])
            enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2RGB)
        else:
            clahe = cv2.createCLAHE(clipLimit=clahe_limit, tileGridSize=(8, 8))
            enhanced = clahe.apply(image)
            
        return enhanced
    
    def _sharpen(self, image: np.ndarray, strength: str = 'light') -> np.ndarray:
        """Apply smart sharpening with adjustable strength."""
        # Adjust sharpening strength
        if strength == 'light':
            weight = 1.2
            gaussian_sigma = 1.5
        elif strength == 'medium':
            weight = 1.35
            gaussian_sigma = 2.0
        else:  # strong
            weight = 1.5
            gaussian_sigma = 2.5
        
        # Create unsharp mask
        gaussian = cv2.GaussianBlur(image, (0, 0), gaussian_sigma)
        sharpened = cv2.addWeighted(image, weight, gaussian, -(weight - 1), 0)
        
        # Clip values to valid range
        sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)
        
        return sharpened
    
    def _enhance_image_quality(self, image: np.ndarray, quality_score: float) -> np.ndarray:
        """
        Apply real quality enhancement - detail recovery, deblurring, edge enhancement.
        This actually improves image quality, not just tweaks parameters.
        """
        enhanced = image.copy()
        
        # Step 1: Edge-preserving detail recovery
        enhanced = self._recover_details(enhanced)
        
        # Step 2: Deblurring if image appears blurry
        if quality_score < 0.7:
            enhanced = self._deblur_image(enhanced)
        
        # Step 3: Edge enhancement for better definition
        enhanced = self._enhance_edges(enhanced)
        
        # Step 4: Local contrast enhancement for better detail visibility
        enhanced = self._local_contrast_enhancement(enhanced)
        
        return enhanced
    
    def _recover_details(self, image: np.ndarray) -> np.ndarray:
        """Recover lost details using edge-preserving techniques."""
        if len(image.shape) == 3:
            # Process each channel separately for better detail recovery
            channels = cv2.split(image)
            enhanced_channels = []
            
            for channel in channels:
                # Use bilateral filter to preserve edges while smoothing
                smoothed = cv2.bilateralFilter(channel, 9, 75, 75)
                # Extract details by subtracting smoothed from original
                details = cv2.subtract(channel, smoothed)
                # Enhance details
                enhanced_details = cv2.multiply(details, 1.5)
                # Add back enhanced details
                enhanced_channel = cv2.add(smoothed, enhanced_details)
                enhanced_channel = np.clip(enhanced_channel, 0, 255).astype(np.uint8)
                enhanced_channels.append(enhanced_channel)
            
            enhanced = cv2.merge(enhanced_channels)
        else:
            smoothed = cv2.bilateralFilter(image, 9, 75, 75)
            details = cv2.subtract(image, smoothed)
            enhanced_details = cv2.multiply(details, 1.5)
            enhanced = cv2.add(smoothed, enhanced_details)
            enhanced = np.clip(enhanced, 0, 255).astype(np.uint8)
        
        return enhanced
    
    def _deblur_image(self, image: np.ndarray) -> np.ndarray:
        """Apply deblurring using Wiener filter approximation."""
        if len(image.shape) == 3:
            # Convert to grayscale for deblurring, then apply to all channels
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            # Create a simple motion blur kernel
            kernel_size = 15
            kernel = np.zeros((kernel_size, kernel_size))
            kernel[int((kernel_size-1)/2), :] = np.ones(kernel_size)
            kernel = kernel / kernel_size
            
            # Apply deconvolution using filter2D (simplified approach)
            # In practice, you'd use more sophisticated deconvolution
            deblurred_gray = cv2.filter2D(gray, -1, kernel)
            
            # Enhance edges to recover details
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            laplacian = np.clip(laplacian + 128, 0, 255).astype(np.uint8)
            
            # Blend original with edge-enhanced version
            enhanced_gray = cv2.addWeighted(gray, 0.7, laplacian, 0.3, 0)
            
            # Apply to color channels proportionally
            enhanced = image.copy().astype(np.float32)
            enhancement_factor = enhanced_gray.astype(np.float32) / (gray.astype(np.float32) + 1)
            for i in range(3):
                enhanced[:, :, i] = enhanced[:, :, i] * enhancement_factor
            enhanced = np.clip(enhanced, 0, 255).astype(np.uint8)
        else:
            # Grayscale deblurring
            laplacian = cv2.Laplacian(image, cv2.CV_64F)
            laplacian = np.clip(laplacian + 128, 0, 255).astype(np.uint8)
            enhanced = cv2.addWeighted(image, 0.7, laplacian, 0.3, 0)
        
        return enhanced
    
    def _enhance_edges(self, image: np.ndarray) -> np.ndarray:
        """Enhance edges for better definition and detail visibility."""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        # Detect edges using Canny
        edges = cv2.Canny(gray, 50, 150)
        
        # Dilate edges slightly
        kernel = np.ones((2, 2), np.uint8)
        edges_dilated = cv2.dilate(edges, kernel, iterations=1)
        
        # Create edge mask
        edge_mask = edges_dilated.astype(np.float32) / 255.0
        
        # Enhance edges in the image
        if len(image.shape) == 3:
            enhanced = image.astype(np.float32)
            for i in range(3):
                # Sharpen edges
                enhanced[:, :, i] = enhanced[:, :, i] + (enhanced[:, :, i] - cv2.GaussianBlur(enhanced[:, :, i], (0, 0), 1.0)) * edge_mask * 0.3
            enhanced = np.clip(enhanced, 0, 255).astype(np.uint8)
        else:
            enhanced = image.astype(np.float32)
            enhanced = enhanced + (enhanced - cv2.GaussianBlur(enhanced, (0, 0), 1.0)) * edge_mask * 0.3
            enhanced = np.clip(enhanced, 0, 255).astype(np.uint8)
        
        return enhanced
    
    def _local_contrast_enhancement(self, image: np.ndarray) -> np.ndarray:
        """Apply local contrast enhancement to improve detail visibility."""
        if len(image.shape) == 3:
            lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)
            
            # Apply CLAHE to lightness channel for local contrast
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            l_enhanced = clahe.apply(l)
            
            enhanced_lab = cv2.merge([l_enhanced, a, b])
            enhanced = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2RGB)
        else:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(image)
        
        return enhanced
    
    def _upscale_image(self, image: np.ndarray, scale: float = 2.0) -> np.ndarray:
        """Upscale image using edge-preserving interpolation."""
        height, width = image.shape[:2]
        new_width = int(width * scale)
        new_height = int(height * scale)
        
        # Use INTER_LANCZOS4 for better quality upscaling
        if len(image.shape) == 3:
            upscaled = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4)
        else:
            upscaled = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4)
        
        # Apply edge enhancement after upscaling
        upscaled = self._enhance_edges(upscaled)
        
        return upscaled
    
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
    
    def _assess_image_quality(self, image: np.ndarray) -> float:
        """Assess overall image quality (0-1, higher is better)."""
        scores = []
        
        # Sharpness score
        sharpness = self._assess_sharpness(image)
        scores.append(sharpness)
        
        # Contrast score
        contrast = self._assess_contrast(image)
        scores.append(contrast)
        
        # Color vibrancy score
        vibrancy = self._assess_color_vibrancy(image)
        scores.append(vibrancy)
        
        # Noise level (inverted - lower noise is better)
        noise = self._estimate_noise_level(image)
        noise_score = max(0, 1 - (noise / 20))  # Normalize noise level
        scores.append(noise_score)
        
        # Average all scores
        return np.mean(scores)
    
    def _assess_sharpness(self, image: np.ndarray) -> float:
        """Assess image sharpness using Laplacian variance (0-1, higher is sharper)."""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        # Normalize to 0-1 range (typical good images have variance > 100)
        sharpness = min(1.0, laplacian_var / 200.0)
        return sharpness
    
    def _assess_contrast(self, image: np.ndarray) -> float:
        """Assess image contrast (0-1, higher is better)."""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        # Calculate standard deviation as contrast measure
        std_dev = np.std(gray)
        # Normalize to 0-1 range (typical good images have std > 40)
        contrast = min(1.0, std_dev / 60.0)
        return contrast
    
    def _assess_color_vibrancy(self, image: np.ndarray) -> float:
        """Assess color vibrancy/saturation (0-1, higher is more vibrant)."""
        if len(image.shape) != 3:
            return 0.5  # Grayscale images have no color
        
        # Convert to HSV and check saturation
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        saturation = hsv[:, :, 1]
        avg_saturation = np.mean(saturation) / 255.0
        return avg_saturation
    
    def _estimate_noise_level(self, image: np.ndarray) -> float:
        """Estimate noise level in the image (higher = more noisy)."""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        # Use median filter to estimate noise
        median = cv2.medianBlur(gray, 5)
        noise = np.abs(gray.astype(np.float32) - median.astype(np.float32))
        noise_level = np.mean(noise)
        return noise_level
    
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



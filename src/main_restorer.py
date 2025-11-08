"""
Main restoration orchestrator that combines all modules.
"""

import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional
import logging

from .restoration_engine import RestorationEngine
from .face_preservation import FacePreservation
from .text_reconstruction import TextReconstruction
from .multi_frame_comparison import MultiFrameComparison
from .report_generator import ReportGenerator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LostDetailRestorer:
    """
    Main class for AI Lost Detail Restorer.
    Orchestrates all restoration modules.
    """
    
    def __init__(self):
        self.restoration_engine = RestorationEngine()
        self.face_preservation = FacePreservation()
        self.text_reconstruction = TextReconstruction()
        self.multi_frame_comparison = MultiFrameComparison()
        self.report_generator = ReportGenerator()
        
    def restore(self, 
               image_path: Optional[str] = None,
               image: Optional[np.ndarray] = None,
               additional_frames: Optional[List[np.ndarray]] = None,
               preserve_faces: bool = True,
               reconstruct_text: bool = True,
               use_multi_frame: bool = True) -> Tuple[np.ndarray, Dict]:
        """
        Main restoration function.
        
        Args:
            image_path: Path to input image (if image not provided)
            image: Input image as numpy array
            additional_frames: Optional additional frames of same scene
            preserve_faces: Whether to preserve face identity
            reconstruct_text: Whether to reconstruct text
            use_multi_frame: Whether to use multi-frame comparison
            
        Returns:
            Restored image and comprehensive report
        """
        # Load image
        if image is None:
            if image_path is None:
                raise ValueError("Either image_path or image must be provided")
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Could not load image from {image_path}")
        
        original_image = image.copy()
        all_frames = [original_image]
        
        # Multi-frame comparison if available
        multi_frame_results = None
        if use_multi_frame and additional_frames:
            all_frames.extend(additional_frames)
            combined_image, multi_frame_results = self.multi_frame_comparison.compare_and_restore(all_frames)
            image = combined_image
        
        # Main restoration
        enhanced_image, restoration_results = self.restoration_engine.enhance_image(
            image,
            denoise=True,
            sharpen=True,
            contrast=True,
            color_correction=True
        )
        
        # Face preservation
        face_results = None
        if preserve_faces:
            enhanced_image, face_results = self.face_preservation.enhance_face_preserving_identity(
                original_image, enhanced_image
            )
        
        # Text reconstruction
        text_results = None
        if reconstruct_text:
            enhanced_image, text_results = self.text_reconstruction.restore_blurred_text(
                original_image, enhanced_image
            )
        
        # Generate comprehensive report
        report = self.report_generator.create_report(
            restoration_results=restoration_results,
            face_results=face_results,
            text_results=text_results,
            multi_frame_results=multi_frame_results
        )
        
        return enhanced_image, report
    
    def restore_faded_photo(self, 
                           image_path: Optional[str] = None,
                           image: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Dict]:
        """
        Specialized restoration for faded photographs.
        
        Args:
            image_path: Path to faded image
            image: Faded image as numpy array
            
        Returns:
            Restored image and report
        """
        if image is None:
            if image_path is None:
                raise ValueError("Either image_path or image must be provided")
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Could not load image from {image_path}")
        
        original_image = image.copy()
        
        # Use specialized faded photo restoration
        restored_image, restoration_results = self.restoration_engine.restore_faded_photo(image)
        
        # Preserve faces if present
        restored_image, face_results = self.face_preservation.enhance_face_preserving_identity(
            original_image, restored_image
        )
        
        # Generate report
        report = self.report_generator.create_report(
            restoration_results=restoration_results,
            face_results=face_results
        )
        
        return restored_image, report
    
    def save_result(self, image: np.ndarray, output_path: str):
        """Save restored image."""
        cv2.imwrite(output_path, image)
        logger.info(f"Restored image saved to {output_path}")



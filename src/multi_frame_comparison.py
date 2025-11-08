"""
Multi-frame comparison module for analyzing multiple frames of the same scene.
"""

import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MultiFrameComparison:
    """Compare and combine multiple frames to recover lost details."""
    
    def __init__(self):
        self.alignment_method = 'ecc'  # Enhanced Correlation Coefficient
        
    def align_frames(self, frames: List[np.ndarray]) -> List[np.ndarray]:
        """
        Align multiple frames to a common reference.
        
        Args:
            frames: List of input frames
            
        Returns:
            List of aligned frames
        """
        if len(frames) < 2:
            return frames
            
        # Use first frame as reference
        reference = frames[0]
        aligned_frames = [reference]
        
        # Convert to grayscale for alignment
        ref_gray = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY) if len(reference.shape) == 3 else reference
        
        for frame in frames[1:]:
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
            
            # Find transformation matrix
            warp_matrix = self._find_alignment(ref_gray, frame_gray)
            
            if warp_matrix is not None:
                # Apply transformation
                h, w = frame.shape[:2]
                aligned = cv2.warpAffine(frame, warp_matrix, (w, h),
                                        flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
                aligned_frames.append(aligned)
            else:
                aligned_frames.append(frame)
                
        return aligned_frames
    
    def _find_alignment(self, reference: np.ndarray, image: np.ndarray) -> Optional[np.ndarray]:
        """
        Find alignment transformation between two images.
        
        Args:
            reference: Reference image
            image: Image to align
            
        Returns:
            Transformation matrix or None
        """
        # Resize if needed
        if reference.shape != image.shape:
            image = cv2.resize(image, (reference.shape[1], reference.shape[0]))
            
        # Use ECC (Enhanced Correlation Coefficient) algorithm
        try:
            # Define initial transformation (identity)
            warp_matrix = np.eye(2, 3, dtype=np.float32)
            
            # Define termination criteria
            criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 50, 0.001)
            
            # Find transformation
            _, warp_matrix = cv2.findTransformECC(
                reference, image, warp_matrix, cv2.MOTION_AFFINE, criteria
            )
            
            return warp_matrix
        except cv2.error as e:
            logger.warning(f"Alignment failed: {e}")
            return None
    
    def combine_frames(self, frames: List[np.ndarray], method: str = 'median') -> np.ndarray:
        """
        Combine multiple aligned frames to recover details.
        
        Args:
            frames: List of aligned frames
            method: Combination method ('median', 'mean', 'max_variance')
            
        Returns:
            Combined image
        """
        if len(frames) == 1:
            return frames[0]
            
        # Stack frames
        frames_array = np.array(frames)
        
        if method == 'median':
            # Median filtering reduces noise and preserves details
            combined = np.median(frames_array, axis=0).astype(np.uint8)
        elif method == 'mean':
            # Mean averaging
            combined = np.mean(frames_array, axis=0).astype(np.uint8)
        elif method == 'max_variance':
            # Use pixel with maximum variance (most detail)
            variance = np.var(frames_array, axis=0)
            max_var_indices = np.argmax(variance, axis=0)
            h, w = frames[0].shape[:2]
            combined = np.zeros_like(frames[0])
            for i in range(h):
                for j in range(w):
                    combined[i, j] = frames_array[max_var_indices[i, j], i, j]
        else:
            combined = np.median(frames_array, axis=0).astype(np.uint8)
            
        return combined
    
    def compare_and_restore(self, frames: List[np.ndarray]) -> Tuple[np.ndarray, Dict]:
        """
        Compare multiple frames and restore lost details.
        
        Args:
            frames: List of input frames (can be same scene from different angles/times)
            
        Returns:
            Restored image and comparison report
        """
        report = {
            'num_frames': len(frames),
            'operations': [],
            'improvements': {}
        }
        
        if len(frames) == 0:
            raise ValueError("No frames provided")
            
        if len(frames) == 1:
            return frames[0], report
        
        # Align frames
        aligned_frames = self.align_frames(frames)
        report['operations'].append('frame_alignment')
        
        # Combine frames
        combined = self.combine_frames(aligned_frames, method='median')
        report['operations'].append('multi_frame_fusion')
        
        # Calculate improvements
        if len(frames) > 1:
            # Compare variance (detail) between first frame and combined
            first_gray = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY) if len(frames[0].shape) == 3 else frames[0]
            combined_gray = cv2.cvtColor(combined, cv2.COLOR_BGR2GRAY) if len(combined.shape) == 3 else combined
            
            first_var = np.var(first_gray)
            combined_var = np.var(combined_gray)
            
            improvement = (combined_var / first_var - 1) * 100 if first_var > 0 else 0
            report['improvements']['detail_recovery'] = f"{improvement:.1f}% increase in detail"
            report['improvements']['noise_reduction'] = f"Reduced noise using {len(frames)} frames"
        
        return combined, report
    
    def detect_motion_regions(self, frames: List[np.ndarray]) -> List[Dict]:
        """
        Detect regions with motion between frames.
        
        Args:
            frames: List of frames
            
        Returns:
            List of motion regions
        """
        if len(frames) < 2:
            return []
            
        motion_regions = []
        
        # Convert to grayscale
        grays = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) if len(f.shape) == 3 else f 
                for f in frames]
        
        # Align frames first
        aligned_grays = self.align_frames(grays)
        
        # Compute difference
        for i in range(1, len(aligned_grays)):
            diff = cv2.absdiff(aligned_grays[0], aligned_grays[i])
            
            # Threshold
            _, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
            
            # Find contours
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                if w * h > 100:  # Filter small regions
                    motion_regions.append({
                        'bbox': (x, y, w, h),
                        'frame_pair': (0, i)
                    })
                    
        return motion_regions


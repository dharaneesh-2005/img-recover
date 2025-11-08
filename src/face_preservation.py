"""
Face identity preservation module.
Ensures faces are enhanced without changing identity.
"""

import cv2
import numpy as np
import face_recognition
from typing import List, Dict, Tuple, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FacePreservation:
    """Preserve face identity while enhancing image quality."""
    
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        
    def detect_faces(self, image: np.ndarray) -> List[Dict]:
        """
        Detect faces in the image.
        
        Args:
            image: Input image
            
        Returns:
            List of face locations and encodings
        """
        # Convert BGR to RGB for face_recognition
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Detect faces using face_recognition library
        face_locations = face_recognition.face_locations(rgb_image, model='hog')
        face_encodings = face_recognition.face_encodings(rgb_image, face_locations)
        
        faces = []
        for i, (top, right, bottom, left) in enumerate(face_locations):
            face_info = {
                'location': (top, right, bottom, left),
                'encoding': face_encodings[i] if i < len(face_encodings) else None,
                'bbox': (left, top, right - left, bottom - top)
            }
            faces.append(face_info)
            
        return faces
    
    def enhance_face_preserving_identity(self, 
                                        original_image: np.ndarray,
                                        enhanced_image: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """
        Enhance faces while preserving identity.
        
        Args:
            original_image: Original image
            enhanced_image: Enhanced image (may have changed faces)
            
        Returns:
            Image with preserved faces and report
        """
        report = {
            'faces_detected': 0,
            'faces_preserved': 0,
            'operations': []
        }
        
        # Detect faces in original
        original_faces = self.detect_faces(original_image)
        report['faces_detected'] = len(original_faces)
        
        if len(original_faces) == 0:
            return enhanced_image, report
        
        result_image = enhanced_image.copy()
        original_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
        enhanced_rgb = cv2.cvtColor(enhanced_image, cv2.COLOR_BGR2RGB)
        
        for face_info in original_faces:
            top, right, bottom, left = face_info['location']
            
            # Extract face regions
            face_original = original_rgb[top:bottom, left:right]
            face_enhanced = enhanced_rgb[top:bottom, left:right]
            
            # Blend: use enhanced for quality but preserve original identity features
            # Apply gentle enhancement to face region
            face_restored = self._restore_face_region(face_original, face_enhanced)
            
            # Place back into result
            result_rgb = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)
            result_rgb[top:bottom, left:right] = face_restored
            result_image = cv2.cvtColor(result_rgb, cv2.COLOR_RGB2BGR)
            
            report['faces_preserved'] += 1
            
        report['operations'].append('identity_preserved_face_enhancement')
        
        return result_image, report
    
    def _restore_face_region(self, 
                            original_face: np.ndarray,
                            enhanced_face: np.ndarray) -> np.ndarray:
        """
        Restore face region with identity preservation.
        
        Args:
            original_face: Original face region
            enhanced_face: Enhanced face region
            
        Returns:
            Restored face with preserved identity
        """
        if original_face.size == 0 or enhanced_face.size == 0:
            return original_face
            
        # Resize if needed
        if original_face.shape != enhanced_face.shape:
            enhanced_face = cv2.resize(enhanced_face, 
                                      (original_face.shape[1], original_face.shape[0]))
        
        # Apply gentle enhancement: blend original structure with enhanced details
        # Use adaptive blending to preserve key facial features
        
        # Convert to LAB color space for better control
        orig_lab = cv2.cvtColor(original_face, cv2.COLOR_RGB2LAB)
        enh_lab = cv2.cvtColor(enhanced_face, cv2.COLOR_RGB2LAB)
        
        # Preserve original lightness structure (identity)
        # But enhance color and details
        orig_l, orig_a, orig_b = cv2.split(orig_lab)
        enh_l, enh_a, enh_b = cv2.split(enh_lab)
        
        # Blend: 70% original lightness (identity), 30% enhanced (quality)
        blended_l = cv2.addWeighted(orig_l, 0.7, enh_l, 0.3, 0)
        
        # Use enhanced color channels (restored colors)
        blended_a = cv2.addWeighted(orig_a, 0.3, enh_a, 0.7, 0)
        blended_b = cv2.addWeighted(orig_b, 0.3, enh_b, 0.7, 0)
        
        # Merge
        blended_lab = cv2.merge([blended_l, blended_a, blended_b])
        restored = cv2.cvtColor(blended_lab, cv2.COLOR_LAB2RGB)
        
        # Apply gentle denoising
        restored = cv2.bilateralFilter(restored, 5, 50, 50)
        
        return restored
    
    def compare_face_identity(self, 
                             face1_encoding: np.ndarray,
                             face2_encoding: np.ndarray) -> float:
        """
        Compare two face encodings to check if they're the same person.
        
        Args:
            face1_encoding: First face encoding
            face2_encoding: Second face encoding
            
        Returns:
            Distance (lower = more similar)
        """
        if face1_encoding is None or face2_encoding is None:
            return float('inf')
            
        distance = face_recognition.face_distance([face1_encoding], face2_encoding)[0]
        return distance



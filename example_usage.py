#!/usr/bin/env python3
"""
Example usage of AI Lost Detail Restorer.
"""

import cv2
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.main_restorer import LostDetailRestorer


def example_basic_restoration():
    """Example: Basic image restoration."""
    print("Example 1: Basic Image Restoration")
    print("-" * 50)
    
    restorer = LostDetailRestorer()
    
    # Restore an image
    restored_image, report = restorer.restore(
        image_path='path/to/your/image.jpg',
        preserve_faces=True,
        reconstruct_text=True
    )
    
    # Save result
    cv2.imwrite('restored_output.jpg', restored_image)
    
    # Print report summary
    print("\nRestoration Report:")
    print(f"Operations: {report['summary']['operations_performed']}")
    print(f"Details Restored: {report['summary']['details_restored']}")
    print("\n")


def example_faded_photo():
    """Example: Restore a faded photograph."""
    print("Example 2: Faded Photo Restoration")
    print("-" * 50)
    
    restorer = LostDetailRestorer()
    
    # Use specialized faded photo restoration
    restored_image, report = restorer.restore_faded_photo(
        image_path='path/to/faded_photo.jpg'
    )
    
    cv2.imwrite('faded_restored.jpg', restored_image)
    print("Faded photo restored and saved!")
    print("\n")


def example_multi_frame():
    """Example: Multi-frame comparison."""
    print("Example 3: Multi-Frame Comparison")
    print("-" * 50)
    
    restorer = LostDetailRestorer()
    
    # Load multiple frames
    frames = [
        cv2.imread('frame1.jpg'),
        cv2.imread('frame2.jpg'),
        cv2.imread('frame3.jpg')
    ]
    
    # Remove None frames
    frames = [f for f in frames if f is not None]
    
    if len(frames) > 0:
        restored_image, report = restorer.restore(
            image=frames[0],
            additional_frames=frames[1:] if len(frames) > 1 else None,
            use_multi_frame=True,
            preserve_faces=True,
            reconstruct_text=True
        )
        
        cv2.imwrite('multi_frame_restored.jpg', restored_image)
        print(f"Combined {len(frames)} frames for restoration!")
        print("\n")


def example_programmatic():
    """Example: Using the API programmatically."""
    print("Example 4: Programmatic Usage")
    print("-" * 50)
    
    restorer = LostDetailRestorer()
    
    # Load image as numpy array
    image = cv2.imread('path/to/image.jpg')
    
    if image is not None:
        # Restore with custom options
        restored, report = restorer.restore(
            image=image,
            preserve_faces=True,
            reconstruct_text=True,
            use_multi_frame=False
        )
        
        # Access detailed report
        if 'detailed_results' in report:
            if 'face_preservation' in report['detailed_results']:
                faces = report['detailed_results']['face_preservation']
                print(f"Faces detected: {faces.get('faces_detected', 0)}")
                print(f"Faces preserved: {faces.get('faces_preserved', 0)}")
            
            if 'text_reconstruction' in report['detailed_results']:
                text_info = report['detailed_results']['text_reconstruction']
                print(f"Text regions found: {text_info.get('text_regions_found', 0)}")
                for text_item in text_info.get('text_detected', []):
                    print(f"  - Detected: '{text_item['text']}'")
        
        # Save result
        cv2.imwrite('programmatic_output.jpg', restored)
        print("\nRestoration complete!")
    else:
        print("Could not load image!")
    print("\n")


if __name__ == '__main__':
    print("=" * 60)
    print("AI Lost Detail Restorer - Usage Examples")
    print("=" * 60)
    print("\nNote: Update the image paths in the examples before running.")
    print("\nUncomment the example you want to run:\n")
    
    # Uncomment the example you want to run:
    # example_basic_restoration()
    # example_faded_photo()
    # example_multi_frame()
    # example_programmatic()
    
    print("Examples are ready! Uncomment the function calls above to run them.")



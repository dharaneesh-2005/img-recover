#!/usr/bin/env python3
"""
Quick test demo of AI Lost Detail Restorer.
Creates a test image and demonstrates restoration.
"""

import cv2
import numpy as np
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.main_restorer import LostDetailRestorer

def create_test_image():
    """Create a simple test image with some details."""
    # Create a 400x400 image
    img = np.zeros((400, 400, 3), dtype=np.uint8)
    
    # Add a gradient background
    for i in range(400):
        intensity = int(255 * (i / 400))
        img[i, :] = [intensity // 3, intensity // 2, intensity]
    
    # Add some shapes (simulating details to restore)
    cv2.rectangle(img, (50, 50), (150, 150), (255, 255, 255), -1)
    cv2.circle(img, (300, 200), 50, (0, 255, 0), -1)
    cv2.putText(img, 'TEST', (100, 250), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    # Add some noise to simulate low quality
    noise = np.random.randint(0, 50, img.shape, dtype=np.uint8)
    img = cv2.add(img, noise)
    
    # Slightly blur to simulate loss of detail
    img = cv2.GaussianBlur(img, (5, 5), 0)
    
    return img

def main():
    print("=" * 60)
    print("AI Lost Detail Restorer - Quick Test Demo")
    print("=" * 60)
    print()
    
    # Create test image
    print("Creating test image...")
    test_image = create_test_image()
    cv2.imwrite('test_input.jpg', test_image)
    print("Test image saved as 'test_input.jpg'")
    print()
    
    # Initialize restorer
    print("Initializing AI Lost Detail Restorer...")
    restorer = LostDetailRestorer()
    print("Restorer initialized!")
    print()
    
    # Restore image
    print("Processing image (this may take a moment)...")
    print("- Denoising...")
    print("- Enhancing details...")
    print("- Applying restoration techniques...")
    
    try:
        restored_image, report = restorer.restore(
            image=test_image,
            preserve_faces=False,  # No faces in test image
            reconstruct_text=True,
            use_multi_frame=False
        )
        
        # Save result
        cv2.imwrite('test_output.jpg', restored_image)
        print()
        print("=" * 60)
        print("Restoration Complete!")
        print("=" * 60)
        print()
        print("Results:")
        print(f"  Input:  test_input.jpg")
        print(f"  Output: test_output.jpg")
        print()
        
        if 'summary' in report:
            summary = report['summary']
            if 'operations_performed' in summary:
                print("Operations performed:")
                for op in summary['operations_performed']:
                    print(f"  • {op}")
            print()
            
            if 'details_restored' in summary:
                print("Details restored:")
                for detail in summary['details_restored']:
                    print(f"  • {detail}")
        
        print()
        print("=" * 60)
        print("Check 'test_output.jpg' to see the restored image!")
        print("=" * 60)
        
    except Exception as e:
        print(f"Error during restoration: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()


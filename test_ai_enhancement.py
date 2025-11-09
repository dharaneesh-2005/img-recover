#!/usr/bin/env python3
"""Test AI enhancement with portable Real-ESRGAN."""

import cv2
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.main_restorer import LostDetailRestorer

def main():
    print("=" * 60)
    print("Testing AI Enhancement with Portable Real-ESRGAN")
    print("=" * 60)
    print()
    
    # Initialize restorer (will auto-detect portable executable)
    print("Initializing AI Lost Detail Restorer...")
    restorer = LostDetailRestorer()
    print("Restorer initialized!")
    print()
    
    # Load test image
    if not os.path.exists('test_input.jpg'):
        print("Error: test_input.jpg not found. Run test_demo.py first.")
        return
    
    print("Loading test image...")
    img = cv2.imread('test_input.jpg')
    if img is None:
        print("Error: Could not load test_input.jpg")
        return
    
    print(f"Image size: {img.shape[1]}x{img.shape[0]}")
    print()
    
    # Restore with AI
    print("Processing with AI enhancement (this may take a moment)...")
    try:
        restored, report = restorer.restore(image=img)
        
        # Save result
        cv2.imwrite('test_ai_output.jpg', restored)
        print()
        print("=" * 60)
        print("AI Enhancement Complete!")
        print("=" * 60)
        print()
        print(f"Input:  test_input.jpg ({img.shape[1]}x{img.shape[0]})")
        print(f"Output: test_ai_output.jpg ({restored.shape[1]}x{restored.shape[0]})")
        print()
        
        if 'summary' in report:
            summary = report['summary']
            if 'operations_performed' in summary:
                print("Operations performed:")
                for op in summary['operations_performed']:
                    print(f"  • {op}")
            print()
            
            if 'improvements' in summary:
                print("Improvements:")
                for key, value in summary.get('improvements', {}).items():
                    print(f"  • {key}: {value}")
        
        print()
        print("=" * 60)
        print("Check 'test_ai_output.jpg' to see the AI-enhanced image!")
        print("=" * 60)
        
    except Exception as e:
        print(f"Error during AI enhancement: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()


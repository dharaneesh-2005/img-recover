#!/usr/bin/env python3
"""
Command-line interface for AI Lost Detail Restorer.
"""

import argparse
import cv2
import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.main_restorer import LostDetailRestorer
from src.report_generator import ReportGenerator


def main():
    parser = argparse.ArgumentParser(
        description='AI Lost Detail Restorer - Restore lost details in old photographs'
    )
    
    parser.add_argument('input', help='Input image path')
    parser.add_argument('-o', '--output', help='Output image path', default=None)
    parser.add_argument('-r', '--report', help='Save restoration report to file', default=None)
    parser.add_argument('--faded', action='store_true', help='Use specialized faded photo restoration')
    parser.add_argument('--no-faces', action='store_true', help='Disable face preservation')
    parser.add_argument('--no-text', action='store_true', help='Disable text reconstruction')
    parser.add_argument('--frames', nargs='+', help='Additional frames for multi-frame comparison')
    
    args = parser.parse_args()
    
    # Check if input file exists
    if not os.path.exists(args.input):
        print(f"Error: Input file '{args.input}' not found.")
        sys.exit(1)
    
    # Determine output path
    if args.output is None:
        input_path = Path(args.input)
        args.output = str(input_path.parent / f"{input_path.stem}_restored{input_path.suffix}")
    
    # Initialize restorer
    print("Initializing AI Lost Detail Restorer...")
    restorer = LostDetailRestorer()
    
    # Load additional frames if provided
    additional_frames = None
    if args.frames:
        print(f"Loading {len(args.frames)} additional frames...")
        additional_frames = []
        for frame_path in args.frames:
            if not os.path.exists(frame_path):
                print(f"Warning: Frame '{frame_path}' not found. Skipping.")
                continue
            frame = cv2.imread(frame_path)
            if frame is not None:
                additional_frames.append(frame)
            else:
                print(f"Warning: Could not load frame '{frame_path}'. Skipping.")
    
    # Restore image
    print(f"Processing image: {args.input}")
    
    try:
        if args.faded:
            print("Using specialized faded photo restoration...")
            restored_image, report = restorer.restore_faded_photo(image_path=args.input)
        else:
            restored_image, report = restorer.restore(
                image_path=args.input,
                additional_frames=additional_frames,
                preserve_faces=not args.no_faces,
                reconstruct_text=not args.no_text,
                use_multi_frame=additional_frames is not None and len(additional_frames) > 0
            )
        
        # Save restored image
        print(f"Saving restored image to: {args.output}")
        restorer.save_result(restored_image, args.output)
        
        # Save report if requested
        if args.report:
            report_gen = ReportGenerator()
            report_format = 'json' if args.report.endswith('.json') else 'txt'
            report_gen.save_report(report, args.report, format=report_format)
            print(f"Restoration report saved to: {args.report}")
        
        # Print summary
        print("\n" + "=" * 60)
        print("Restoration Complete!")
        print("=" * 60)
        if 'summary' in report:
            summary = report['summary']
            if 'operations_performed' in summary:
                print(f"\nOperations: {', '.join(summary['operations_performed'])}")
            if 'details_restored' in summary:
                print(f"\nDetails Restored:")
                for detail in summary['details_restored']:
                    print(f"  • {detail}")
        print("=" * 60)
        
    except Exception as e:
        print(f"Error during restoration: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()



"""
Main script for invoice extraction pipeline.
Processes tractor invoices and extracts key information using YOLO and VLM models.
"""

import os
import sys
import argparse
import glob
import json
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm

from utils.extractor import InvoiceExtractor
from utils.utilities import create_directories, get_device_info

"""
Main script for invoice extraction pipeline.
Processes a single tractor invoice and extracts key information using YOLO and VLM models.
"""


def parse_args():
    parser = argparse.ArgumentParser(description='Tractor Invoice Extraction Pipeline')
    parser.add_argument(
        'image_path',
        type=str,
        help='Path to the invoice image to process'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='output',
        help='Directory to save JSON output (default: output/)'
    )
    parser.add_argument(
        '--save_annotated',
        action='store_true',
        help='Save annotated image with detections'
    )
    parser.add_argument(
        '--annotated_dir',
        type=str,
        default='output/annotated',
        help='Directory to save annotated images (default: output/annotated/)'
    )
    parser.add_argument(
        '--yolo_path',
        type=str,
        default='utils/best.pt',
        help='Path to YOLO model weights'
    )
    parser.add_argument(
        '--vlm_model',
        type=str,
        default='Qwen/Qwen2.5-VL-3B-Instruct',
        help='Hugging Face model name for VLM'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='auto',
        choices=['auto', 'cuda', 'mps', 'cpu'],
        help='Device to use for inference (auto will select best available)'
    )
    parser.add_argument(
        '--show_image',
        action='store_true',
        help='Display annotated image after processing'
    )
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Check if image exists
    if not os.path.exists(args.image_path):
        print(f"❌ Error: Image not found at '{args.image_path}'")
        sys.exit(1)
    
    # Display device information
    device_info = get_device_info()
    print("=" * 60)
    print("🖥️  DEVICE INFORMATION")
    print("=" * 60)
    for key, value in device_info.items():
        print(f"{key}: {value}")
    print("=" * 60)
    print()
    
    # Create output directories
    output_dirs = [args.output_dir]
    if args.save_annotated:
        output_dirs.append(args.annotated_dir)
    create_directories(output_dirs)
    
    # Check if YOLO model exists
    if not os.path.exists(args.yolo_path):
        print(f"❌ Error: YOLO model not found at {args.yolo_path}")
        print(f"Please place your trained YOLO model at {args.yolo_path}")
        sys.exit(1)
    
    # Initialize extractor
    print("⏳ Initializing Invoice Extractor...")
    extractor = InvoiceExtractor(
        yolo_path=args.yolo_path,
        vlm_model_name=args.vlm_model,
        device=args.device
    )
    print()
    
    # Get base filename
    base_name = os.path.splitext(os.path.basename(args.image_path))[0]
    
    print(f"🚀 Processing: {args.image_path}")
    print()
    
    try:
        # Run extraction
        result_json = extractor.run(args.image_path)
        
        # Print result to terminal
        print("=" * 60)
        print("📄 EXTRACTION RESULT")
        print("=" * 60)
        print(json.dumps(result_json, indent=2, ensure_ascii=False))
        print("=" * 60)
        print()
        
        # Save JSON to file
        json_filename = f"{base_name}.json"
        json_path = os.path.join(args.output_dir, json_filename)
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(result_json, f, indent=2, ensure_ascii=False)
        
        print(f"✅ JSON saved to: {json_path}")
        
        # Generate and save annotated image if requested
        if args.save_annotated or args.show_image:
            yolo_results = extractor.yolo(args.image_path, verbose=False)[0]
            annotated_img = yolo_results.plot(labels=True, conf=True)
            
            if args.save_annotated:
                annotated_filename = f"{base_name}_annotated.jpg"
                annotated_path = os.path.join(args.annotated_dir, annotated_filename)
                cv2.imwrite(annotated_path, annotated_img)
                print(f"✅ Annotated image saved to: {annotated_path}")
            
            if args.show_image:
                plt.figure(figsize=(12, 8))
                plt.imshow(cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB))
                plt.title(f"Annotated: {base_name}")
                plt.axis('off')
                plt.tight_layout()
                plt.show()
                plt.close()
        
        print()
        print("=" * 60)
        print("🎉 Processing Complete!")
        print("=" * 60)
        
        return result_json
        
    except Exception as e:
        print(f"❌ Error processing {args.image_path}: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
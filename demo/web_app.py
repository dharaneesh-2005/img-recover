"""
Web-based demo interface for AI Lost Detail Restorer.
Creates a magical, interactive experience.
"""

from flask import Flask, render_template, request, jsonify, send_file
from flask_cors import CORS
import cv2
import numpy as np
import os
import base64
import io
from PIL import Image
import sys

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.main_restorer import LostDetailRestorer

app = Flask(__name__)
CORS(app)

# Initialize restorer
restorer = LostDetailRestorer()

# Create uploads directory
UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'outputs'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)


def base64_to_image(base64_string):
    """Convert base64 string to numpy array."""
    # Remove data URL prefix if present
    if ',' in base64_string:
        base64_string = base64_string.split(',')[1]
    
    # Decode base64
    image_data = base64.b64decode(base64_string)
    
    # Convert to PIL Image
    pil_image = Image.open(io.BytesIO(image_data))
    
    # Convert to RGB if needed
    if pil_image.mode != 'RGB':
        pil_image = pil_image.convert('RGB')
    
    # Convert to numpy array
    image_array = np.array(pil_image)
    
    # Convert RGB to BGR for OpenCV
    image_bgr = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
    
    return image_bgr


def image_to_base64(image):
    """Convert numpy array to base64 string."""
    # Convert BGR to RGB
    if len(image.shape) == 3:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        image_rgb = image
    
    # Convert to PIL Image
    pil_image = Image.fromarray(image_rgb)
    
    # Convert to base64
    buffer = io.BytesIO()
    pil_image.save(buffer, format='PNG')
    image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    
    return image_base64


@app.route('/')
def index():
    """Main page."""
    return render_template('index.html')


@app.route('/api/restore', methods=['POST'])
def restore():
    """Restore image endpoint."""
    try:
        data = request.json
        
        if 'image' not in data:
            return jsonify({'error': 'No image provided'}), 400
        
        # Decode image
        input_image = base64_to_image(data['image'])
        
        # Get options
        preserve_faces = data.get('preserve_faces', True)
        reconstruct_text = data.get('reconstruct_text', True)
        use_multi_frame = data.get('use_multi_frame', False)
        additional_frames = []
        
        # Process additional frames if provided
        if use_multi_frame and 'additional_frames' in data:
            for frame_data in data['additional_frames']:
                additional_frames.append(base64_to_image(frame_data))
        
        # Restore image
        restored_image, report = restorer.restore(
            image=input_image,
            additional_frames=additional_frames if additional_frames else None,
            preserve_faces=preserve_faces,
            reconstruct_text=reconstruct_text,
            use_multi_frame=use_multi_frame and len(additional_frames) > 0
        )
        
        # Convert to base64
        restored_base64 = image_to_base64(restored_image)
        
        return jsonify({
            'success': True,
            'restored_image': restored_base64,
            'report': report
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/restore-faded', methods=['POST'])
def restore_faded():
    """Restore faded photo endpoint."""
    try:
        data = request.json
        
        if 'image' not in data:
            return jsonify({'error': 'No image provided'}), 400
        
        # Decode image
        input_image = base64_to_image(data['image'])
        
        # Restore faded photo
        restored_image, report = restorer.restore_faded_photo(image=input_image)
        
        # Convert to base64
        restored_base64 = image_to_base64(restored_image)
        
        return jsonify({
            'success': True,
            'restored_image': restored_base64,
            'report': report
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    print("=" * 60)
    print("AI Lost Detail Restorer - Web Demo")
    print("=" * 60)
    print("\nStarting server...")
    print("Open your browser to: http://localhost:5000")
    print("\nPress Ctrl+C to stop the server")
    print("=" * 60)
    
    app.run(debug=True, host='0.0.0.0', port=5000)



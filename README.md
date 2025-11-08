# ✨ AI Lost Detail Restorer ✨

A powerful AI-powered system for restoring lost details in old, low-quality photographs. Unlike standard image enhancement tools, this system:

- **Reconstructs blurred text** on signboards
- **Enhances faces** without changing identity
- **Recovers faded photographs** with advanced color restoration
- **Compares multiple frames** (if available) for better detail recovery
- **Produces detailed reports** of what was restored and why

## 🎯 Features

### Core Capabilities

1. **Image Enhancement**
   - Advanced denoising using non-local means
   - Smart sharpening for detail recovery
   - Contrast enhancement with CLAHE
   - Color correction and restoration

2. **Face Identity Preservation**
   - Detects faces in images
   - Enhances face quality while preserving identity
   - Uses advanced blending techniques to maintain facial features

3. **Text Reconstruction**
   - Detects text regions in images
   - Enhances text regions for better OCR
   - Reconstructs blurred/faded text on signboards
   - Uses EasyOCR and Tesseract for robust text detection

4. **Multi-Frame Comparison**
   - Aligns multiple frames of the same scene
   - Combines frames to recover lost details
   - Reduces noise through frame averaging
   - Detects motion regions

5. **Comprehensive Reporting**
   - Detailed reports of all operations performed
   - Quantitative improvement metrics
   - Human-readable summaries
   - JSON and text format support

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- Tesseract OCR (for text reconstruction)
  - Windows: Download from [GitHub](https://github.com/UB-Mannheim/tesseract/wiki)
  - macOS: `brew install tesseract`
  - Linux: `sudo apt-get install tesseract-ocr`

### Install Dependencies

```bash
pip install -r requirements.txt
```

**Note:** Some dependencies (like `dlib` and `face-recognition`) may require additional system libraries. On Linux, you may need:

```bash
sudo apt-get install build-essential cmake libopenblas-dev liblapack-dev
```

## 📖 Usage

### Web Interface (Recommended for Demo)

Start the web server:

```bash
cd demo
python web_app.py
```

Then open your browser to `http://localhost:5000` for a magical, interactive experience!

### Command Line Interface

Basic usage:

```bash
python cli.py input_image.jpg -o output_image.jpg
```

Restore a faded photo:

```bash
python cli.py old_photo.jpg --faded -o restored.jpg
```

With additional frames for multi-frame comparison:

```bash
python cli.py frame1.jpg --frames frame2.jpg frame3.jpg -o combined.jpg
```

Generate a restoration report:

```bash
python cli.py input.jpg -o output.jpg -r report.json
```

Disable specific features:

```bash
python cli.py input.jpg --no-faces --no-text -o output.jpg
```

### Python API

```python
from src.main_restorer import LostDetailRestorer
import cv2

# Initialize restorer
restorer = LostDetailRestorer()

# Restore image
restored_image, report = restorer.restore(
    image_path='old_photo.jpg',
    preserve_faces=True,
    reconstruct_text=True
)

# Save result
cv2.imwrite('restored.jpg', restored_image)

# Access report
print(report['summary'])
```

## 🎨 Use Cases

### 1. Reconstructing Blurred Text on Signboards

Perfect for:
- Old street photos with store signs
- Historical documents
- License plates in old photos

```python
restored, report = restorer.restore(
    image_path='blurry_sign.jpg',
    reconstruct_text=True
)
```

### 2. Enhancing Faces Without Changing Identity

Ideal for:
- Family photo restoration
- Portrait enhancement
- Historical figure photos

```python
restored, report = restorer.restore(
    image_path='old_portrait.jpg',
    preserve_faces=True
)
```

### 3. Recovering Faded Photographs

Specialized restoration for:
- Sun-faded photos
- Old color photographs
- Vintage family albums

```python
restored, report = restorer.restore_faded_photo(
    image_path='faded_photo.jpg'
)
```

### 4. Multi-Frame Comparison

When you have multiple shots of the same scene:
- Different angles of the same subject
- Multiple frames from a video
- Burst photos

```python
frames = [cv2.imread(f'frame{i}.jpg') for i in range(1, 4)]
restored, report = restorer.restore(
    image=frames[0],
    additional_frames=frames[1:],
    use_multi_frame=True
)
```

## 📊 Report Format

The system generates detailed reports in JSON or text format:

```json
{
  "timestamp": "2024-01-15T10:30:00",
  "summary": {
    "operations_performed": [
      "denoising",
      "color_correction",
      "contrast_enhancement",
      "sharpening",
      "identity_preserved_face_enhancement",
      "text_reconstruction"
    ],
    "improvements_made": {
      "noise_reduction": "Applied non-local means denoising",
      "color_restoration": "Enhanced faded colors",
      "detail_increase": "15.3%"
    },
    "details_restored": [
      "Preserved identity of 2 face(s)",
      "Reconstructed 3 text region(s)"
    ]
  }
}
```

## 🏗️ Architecture

```
src/
├── main_restorer.py          # Main orchestrator
├── restoration_engine.py    # Core enhancement engine
├── face_preservation.py      # Face identity preservation
├── text_reconstruction.py    # Text detection and reconstruction
├── multi_frame_comparison.py # Multi-frame analysis
└── report_generator.py       # Report generation

demo/
├── web_app.py                # Flask web interface
└── templates/
    └── index.html            # Web UI
```

## 🔧 Technical Details

### Image Processing Techniques

- **Denoising**: Non-local means denoising for preserving details
- **Sharpening**: Unsharp masking with adaptive parameters
- **Contrast**: CLAHE (Contrast Limited Adaptive Histogram Equalization)
- **Color**: LAB color space manipulation for natural color restoration

### Face Preservation

- Uses `face_recognition` library for face detection
- LAB color space blending to preserve identity
- Adaptive blending ratios (70% original structure, 30% enhanced details)

### Text Reconstruction

- Region detection using adaptive thresholding
- Multi-scale enhancement for OCR
- Dual OCR engines (EasyOCR + Tesseract) for robustness
- Morphological operations for text cleanup

### Multi-Frame Comparison

- ECC (Enhanced Correlation Coefficient) alignment
- Median filtering for noise reduction
- Variance-based detail selection

## 🎯 Performance Tips

1. **For best results with faces**: Ensure faces are clearly visible (even if low quality)
2. **For text reconstruction**: Higher resolution images work better
3. **For multi-frame**: Frames should be of the same scene (alignment helps)
4. **Processing time**: Depends on image size and features enabled
   - Small images (< 1MP): ~2-5 seconds
   - Medium images (1-5MP): ~5-15 seconds
   - Large images (> 5MP): ~15-60 seconds

## 🐛 Troubleshooting

### dlib/face-recognition installation issues

If you encounter issues installing `dlib`:

```bash
# On macOS
brew install cmake
pip install dlib

# On Linux
sudo apt-get install cmake libopenblas-dev liblapack-dev
pip install dlib
```

### Tesseract not found

Make sure Tesseract is installed and in your PATH. You may need to specify the path:

```python
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  # Windows
```

### Out of memory errors

For very large images, consider resizing before processing:

```python
import cv2
image = cv2.imread('large_image.jpg')
image = cv2.resize(image, (2000, 2000))  # Resize to max 2000px
```

## 📝 License

This project is provided as-is for demonstration and educational purposes.

## 🙏 Acknowledgments

- OpenCV for image processing
- face_recognition library for face detection
- EasyOCR and Tesseract for text recognition
- Flask for web interface

## 🚧 Future Enhancements

- [ ] GPU acceleration support
- [ ] Deep learning-based super-resolution
- [ ] Batch processing mode
- [ ] Video restoration support
- [ ] Cloud deployment options

---

**Made with ✨ for preserving memories and restoring lost details**



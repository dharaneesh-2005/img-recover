# AI Models for Image Enhancement Guide

## Overview

The system now supports AI-powered image enhancement using deep learning models, which provide **much better quality improvements** than traditional computer vision methods.

## Available Models

### 1. Real-ESRGAN (Recommended) ⭐

**Real-ESRGAN** is a state-of-the-art image super-resolution and enhancement model that can:
- Upscale images 2x or 4x with high quality
- Recover fine details lost in low-quality images
- Enhance textures and edges naturally
- Reduce artifacts and noise

**Installation:**
```bash
pip install realesrgan
```

**Usage:**
```python
from src.main_restorer import LostDetailRestorer

# Use Real-ESRGAN with 4x upscaling
restorer = LostDetailRestorer(
    use_ai_model=True,
    ai_model_name='realesrgan',
    ai_model_scale=4  # 2 or 4
)

# Restore image
restored, report = restorer.restore(image_path='your_image.jpg')
```

### 2. Traditional Methods (Fallback)

If AI models are not available, the system automatically falls back to traditional computer vision methods.

## Comparison

| Method | Quality | Speed | GPU Required |
|--------|---------|-------|--------------|
| **Real-ESRGAN** | ⭐⭐⭐⭐⭐ Excellent | Medium | Optional (faster with GPU) |
| Traditional CV | ⭐⭐⭐ Good | Fast | No |

## Installation

### Basic Installation (Traditional Methods Only)
```bash
pip install -r requirements.txt
```

### With AI Enhancement (Recommended)
```bash
pip install -r requirements.txt
pip install realesrgan
```

**Note:** Real-ESRGAN will download the model (~67MB) on first use. This is automatic and the model will be cached.

## Usage Examples

### Example 1: Using AI Model (Default)
```python
from src.main_restorer import LostDetailRestorer
import cv2

# Initialize with AI model (default)
restorer = LostDetailRestorer()

# Restore image
restored, report = restorer.restore(image_path='old_photo.jpg')
cv2.imwrite('enhanced.jpg', restored)

print(f"Model used: {report['summary'].get('ai_model', 'Traditional')}")
```

### Example 2: Using Traditional Methods Only
```python
# Disable AI models
restorer = LostDetailRestorer(use_ai_model=False)

restored, report = restorer.restore(image_path='old_photo.jpg')
```

### Example 3: Custom AI Model Settings
```python
# Use 2x upscaling instead of 4x (faster, less memory)
restorer = LostDetailRestorer(
    use_ai_model=True,
    ai_model_name='realesrgan',
    ai_model_scale=2  # 2x instead of 4x
)
```

### Example 4: Command Line with AI
```bash
# The CLI automatically uses AI models if available
python cli.py input.jpg -o output.jpg
```

## Performance Tips

1. **GPU Acceleration**: If you have a CUDA-compatible GPU, Real-ESRGAN will automatically use it for faster processing.

2. **Memory Usage**: 
   - 4x upscaling uses more memory than 2x
   - For very large images, consider using 2x upscaling

3. **Processing Time**:
   - Small images (< 1MP): ~5-15 seconds with AI
   - Medium images (1-5MP): ~15-60 seconds with AI
   - Large images (> 5MP): ~1-5 minutes with AI

4. **First Run**: The first time you use Real-ESRGAN, it will download the model (~67MB). Subsequent runs will be faster.

## Troubleshooting

### Real-ESRGAN Not Found
```bash
pip install realesrgan
```

### Out of Memory Errors
- Use 2x upscaling instead of 4x
- Process smaller images
- Use traditional methods: `use_ai_model=False`

### Model Download Issues
- Check your internet connection
- The model downloads automatically on first use
- Model is cached in `~/.cache/realesrgan/` (Linux/Mac) or `%USERPROFILE%\.cache\realesrgan\` (Windows)

## Web Interface

The web interface at `http://localhost:5000` automatically uses AI models if available. No configuration needed!

## Technical Details

- **Real-ESRGAN**: Uses ESRGAN architecture with improved training
- **Model Size**: ~67MB (downloaded automatically)
- **Supported Formats**: JPEG, PNG, BMP, TIFF
- **Color Spaces**: RGB, BGR (automatic conversion)

## Future Models

Planned support for:
- ESRGAN (alternative model)
- Custom trained models
- Specialized models for faces, text, etc.

---

**For best results, use Real-ESRGAN!** It provides significantly better quality than traditional methods.


# Quick Start Guide

Get started with AI Lost Detail Restorer in minutes!

## 🚀 Fast Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

**Important:** You also need Tesseract OCR installed on your system:
- **Windows**: Download from [GitHub](https://github.com/UB-Mannheim/tesseract/wiki)
- **macOS**: `brew install tesseract`
- **Linux**: `sudo apt-get install tesseract-ocr`

### 2. Try the Web Demo (Easiest!)

```bash
cd demo
python web_app.py
```

Then open `http://localhost:5000` in your browser and upload an image!

### 3. Or Use Command Line

```bash
python cli.py your_image.jpg -o restored.jpg
```

## 📸 Example Use Cases

### Restore a Faded Photo

```bash
python cli.py old_faded_photo.jpg --faded -o restored.jpg
```

### Restore with Text Detection

```bash
python cli.py blurry_sign.jpg -o clear_sign.jpg -r report.json
```

### Multi-Frame Restoration

If you have multiple photos of the same scene:

```bash
python cli.py frame1.jpg --frames frame2.jpg frame3.jpg -o combined.jpg
```

## 🎯 What Gets Restored?

- ✅ **Faces**: Enhanced while preserving identity
- ✅ **Text**: Reconstructed from blurry signboards
- ✅ **Colors**: Restored from faded photos
- ✅ **Details**: Recovered through advanced enhancement
- ✅ **Noise**: Reduced while preserving details

## 📊 View the Report

Every restoration generates a detailed report:

```bash
python cli.py image.jpg -o output.jpg -r report.json
```

The report shows:
- What operations were performed
- What details were restored
- Quantitative improvements
- Detected text and faces

## 🐛 Troubleshooting

### "dlib not found" error

```bash
# macOS
brew install cmake
pip install dlib

# Linux
sudo apt-get install cmake libopenblas-dev liblapack-dev
pip install dlib
```

### "Tesseract not found" error

Make sure Tesseract is installed and in your PATH. On Windows, you may need to add it manually.

### Out of memory?

For very large images, resize them first or process in smaller batches.

## 💡 Tips for Best Results

1. **Higher resolution = Better results** (even if the image is blurry)
2. **Multiple frames** help recover more details
3. **Face preservation** works best when faces are clearly visible
4. **Text reconstruction** works better with clear text regions

## 🎨 Next Steps

- Check out `example_usage.py` for Python API examples
- Read `README.md` for detailed documentation
- Try the web interface for the best visual experience!

---

**Happy Restoring! ✨**



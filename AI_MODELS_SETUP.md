# AI Models Setup Guide

## Current Status

✅ **AI model support has been added to the codebase**
⚠️ **Some AI packages have installation/compatibility issues on Windows**

## Installation Issues

### Issue 1: `realesrgan` package
- **Problem**: Requires `basicsr` which fails to build on Windows
- **Error**: `KeyError: '__version__'` during build

### Issue 2: `py-real-esrgan` package  
- **Problem**: Uses outdated `huggingface_hub` API
- **Error**: `cannot import name 'cached_download'` (should be `hf_hub_download`)

## Solutions

### Option 1: Use Traditional Methods (Current Default)
The system automatically falls back to traditional computer vision methods, which still provide good results:
- Detail recovery
- Edge enhancement  
- Local contrast enhancement
- Deblurring

**No installation needed** - works out of the box!

### Option 2: Fix py-real-esrgan (Advanced)
If you want to use AI models, you can manually fix the compatibility issue:

1. Edit the py-real-esrgan package file:
   ```
   C:\Users\<your_username>\miniconda3\Lib\site-packages\py_real_esrgan\model.py
   ```

2. Change line 6 from:
   ```python
   from huggingface_hub import hf_hub_url, cached_download
   ```
   To:
   ```python
   from huggingface_hub import hf_hub_url, hf_hub_download as cached_download
   ```

3. Then the AI models should work!

### Option 3: Use Linux/WSL
The `realesrgan` package installs more easily on Linux. You could:
- Use WSL (Windows Subsystem for Linux)
- Use a Linux VM
- Use a cloud instance

## Current System Behavior

The system is **smart** and will:
1. ✅ Try to use AI models if available
2. ✅ Automatically fall back to traditional methods if AI models aren't available
3. ✅ Still provide quality enhancements using advanced CV techniques

## Testing

You can test if AI models are available:
```python
from src.main_restorer import LostDetailRestorer

restorer = LostDetailRestorer()
# Check logs - it will tell you if AI models are available
restored, report = restorer.restore(image_path='test.jpg')
print(report['summary'].get('ai_model', 'Using traditional methods'))
```

## Future Improvements

We're working on:
- Better Windows compatibility
- Alternative AI model packages
- Pre-compiled wheels for easier installation

## Recommendation

**For now**: Use the traditional methods - they're already integrated and working well!

The traditional enhancement pipeline includes:
- ✅ Detail recovery algorithms
- ✅ Edge enhancement
- ✅ Local contrast enhancement  
- ✅ Deblurring
- ✅ Adaptive processing

These provide **real quality improvements**, not just parameter tweaks.

---

**The system works great with traditional methods!** AI models are a bonus when they work, but not required for quality enhancement.


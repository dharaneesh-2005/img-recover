"""
AI-powered image enhancement using deep learning models.
Supports multiple models: Real-ESRGAN, ESRGAN, and others.
"""

import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try to import AI enhancement libraries
REAL_ESRGAN_AVAILABLE = False
PY_REAL_ESRGAN_AVAILABLE = False
REAL_ESRGAN_EXE_AVAILABLE = False
REAL_ESRGAN_EXE_PATH = None
ESRGAN_AVAILABLE = False
BASICSR_AVAILABLE = False

# Check for portable Real-ESRGAN executable (Windows)
import os
import subprocess
from pathlib import Path

# Common locations for portable Real-ESRGAN
possible_paths = [
    r"C:\Users\skdha\Downloads\realesgran\realesrgan-ncnn-vulkan.exe",
    os.path.join(os.path.dirname(__file__), "..", "..", "realesgran", "realesrgan-ncnn-vulkan.exe"),
    os.path.join(os.path.dirname(__file__), "..", "realesgran", "realesrgan-ncnn-vulkan.exe"),
]

for exe_path in possible_paths:
    if os.path.exists(exe_path):
        REAL_ESRGAN_EXE_PATH = exe_path
        REAL_ESRGAN_EXE_AVAILABLE = True
        logger.info(f"Found portable Real-ESRGAN executable at: {exe_path}")
        break

# Try py-real-esrgan first (simpler, easier to install on Windows)
try:
    from py_real_esrgan.model import RealESRGAN
    import torch
    PY_REAL_ESRGAN_AVAILABLE = True
    logger.info("py-real-esrgan is available")
except ImportError as e:
    PY_REAL_ESRGAN_AVAILABLE = False
    logger.debug(f"py-real-esrgan not available: {e}")

# Try original realesrgan package
try:
    from realesrgan import RealESRGANer
    REAL_ESRGAN_AVAILABLE = True
    logger.info("Real-ESRGAN is available")
except ImportError:
    try:
        # Try alternative import paths
        try:
            from basicsr.archs.rrdbnet_arch import RRDBNet
            from realesrgan import RealESRGANer
            REAL_ESRGAN_AVAILABLE = True
            logger.info("Real-ESRGAN is available (with BasicSR)")
        except:
            import realesrgan
            REAL_ESRGAN_AVAILABLE = True
            logger.info("Real-ESRGAN is available (alternative import)")
    except ImportError:
        REAL_ESRGAN_AVAILABLE = False
        if not PY_REAL_ESRGAN_AVAILABLE:
            logger.warning("Real-ESRGAN not available. Try: pip install py-real-esrgan")

try:
    from basicsr.archs.rrdbnet_arch import RRDBNet
    BASICSR_AVAILABLE = True
except ImportError:
    pass


class AIEnhancement:
    """
    AI-powered image enhancement using deep learning models.
    Provides much better quality improvements than traditional CV methods.
    """
    
    def __init__(self, model_name: str = 'realesrgan', model_scale: int = 4, exe_path: Optional[str] = None):
        """
        Initialize AI enhancement.
        
        Args:
            model_name: Model to use ('realesrgan', 'esrgan', 'traditional')
            model_scale: Upscaling factor (2 or 4 for Real-ESRGAN)
            exe_path: Path to portable Real-ESRGAN executable (optional)
        """
        self.model_name = model_name
        self.model_scale = model_scale
        self.upsampler = None
        self.py_realesrgan_model = None
        self.exe_path = exe_path or REAL_ESRGAN_EXE_PATH
        self.models_dir = None
        
        # Determine which Real-ESRGAN implementation to use
        if model_name == 'realesrgan':
            # Priority: 1. Portable executable, 2. py-real-esrgan, 3. original package
            if self.exe_path and os.path.exists(self.exe_path):
                self._init_realesrgan_exe()
            elif PY_REAL_ESRGAN_AVAILABLE:
                self._init_py_realesrgan()
            elif REAL_ESRGAN_AVAILABLE:
                self._init_realesrgan()
            else:
                logger.warning(f"Real-ESRGAN not available. Try: pip install py-real-esrgan or provide executable path")
                self.model_name = 'traditional'
        elif model_name == 'esrgan' and BASICSR_AVAILABLE:
            self._init_esrgan()
        else:
            if model_name == 'realesrgan':
                logger.warning(f"Real-ESRGAN not available. Try: pip install py-real-esrgan")
            else:
                logger.warning(f"AI model '{model_name}' not available. Using traditional methods.")
            self.model_name = 'traditional'
    
    def _init_realesrgan(self):
        """Initialize Real-ESRGAN model."""
        try:
            # Real-ESRGAN model paths
            if self.model_scale == 2:
                model_path = 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth'
            elif self.model_scale == 4:
                model_path = 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth'
            else:
                model_path = 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth'
                self.model_scale = 4
            
            # Initialize Real-ESRGAN
            # Note: This will download the model on first use (can take a while)
            # The model will be cached in ~/.cache/realesrgan/ or similar
            self.upsampler = RealESRGANer(
                scale=self.model_scale,
                model_path=model_path,
                model=None,
                tile=0,  # Set tile size for large images (0 = no tiling, set to 400 for large images)
                tile_pad=10,
                pre_pad=0,
                half=False  # Set to True if using GPU with CUDA for faster processing
            )
            logger.info(f"Real-ESRGAN initialized with {self.model_scale}x upscaling")
            logger.info("Note: Model will be downloaded on first use if not already cached")
        except Exception as e:
            logger.error(f"Failed to initialize Real-ESRGAN: {e}")
            logger.info("You may need to install: pip install realesrgan")
            self.model_name = 'traditional'
            self.upsampler = None
    
    def _init_py_realesrgan(self):
        """Initialize py-real-esrgan model (simpler alternative)."""
        try:
            import torch
            from py_real_esrgan.model import RealESRGAN
            from PIL import Image
            
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            logger.info(f"Using device: {device}")
            
            self.py_realesrgan_model = RealESRGAN(device, scale=self.model_scale)
            # Model will be downloaded automatically on first use
            model_name = f'RealESRGAN_x{self.model_scale}.pth'
            self.py_realesrgan_model.load_weights(f'weights/{model_name}', download=True)
            
            logger.info(f"py-real-esrgan initialized with {self.model_scale}x upscaling")
            logger.info("Note: Model will be downloaded on first use if not already cached")
        except Exception as e:
            logger.error(f"Failed to initialize py-real-esrgan: {e}")
            logger.info("You may need to install: pip install py-real-esrgan")
            self.model_name = 'traditional'
            self.py_realesrgan_model = None
    
    def _init_realesrgan_exe(self):
        """Initialize portable Real-ESRGAN executable."""
        try:
            if not os.path.exists(self.exe_path):
                raise FileNotFoundError(f"Real-ESRGAN executable not found at: {self.exe_path}")
            
            # Find models directory
            exe_dir = os.path.dirname(self.exe_path)
            models_dir = os.path.join(exe_dir, 'models')
            
            if not os.path.exists(models_dir):
                raise FileNotFoundError(f"Models directory not found at: {models_dir}")
            
            self.models_dir = models_dir
            
            # Determine model name based on scale
            # Available models: realesrgan-x4plus, realesr-animevideov3-x2/x3/x4
            if self.model_scale == 2:
                self.model_name_exe = 'realesr-animevideov3-x2'
            elif self.model_scale == 3:
                self.model_name_exe = 'realesr-animevideov3-x3'
            elif self.model_scale == 4:
                self.model_name_exe = 'realesrgan-x4plus'  # General purpose model
            else:
                self.model_name_exe = 'realesrgan-x4plus'
                self.model_scale = 4
            
            # Check if model files exist
            model_bin = os.path.join(models_dir, f'{self.model_name_exe}.bin')
            model_param = os.path.join(models_dir, f'{self.model_name_exe}.param')
            
            if not (os.path.exists(model_bin) and os.path.exists(model_param)):
                logger.warning(f"Model {self.model_name_exe} not found, trying alternative...")
                # Try alternative models
                if self.model_scale == 4:
                    self.model_name_exe = 'realesr-animevideov3-x4'
                    model_bin = os.path.join(models_dir, f'{self.model_name_exe}.bin')
                    model_param = os.path.join(models_dir, f'{self.model_name_exe}.param')
            
            if not (os.path.exists(model_bin) and os.path.exists(model_param)):
                raise FileNotFoundError(f"Model files not found: {model_bin} or {model_param}")
            
            logger.info(f"Portable Real-ESRGAN initialized: {self.exe_path}")
            logger.info(f"Using model: {self.model_name_exe} ({self.model_scale}x)")
            
        except Exception as e:
            logger.error(f"Failed to initialize portable Real-ESRGAN: {e}")
            self.model_name = 'traditional'
            self.exe_path = None
    
    def _init_esrgan(self):
        """Initialize ESRGAN model."""
        # ESRGAN initialization would go here
        logger.info("ESRGAN initialization not yet implemented")
        self.model_name = 'traditional'
    
    def enhance(self, image: np.ndarray, upscale: bool = True) -> Tuple[np.ndarray, Dict]:
        """
        Note: For portable executable, upscale parameter controls whether to apply
        the scale factor. The executable always applies the model's scale (2x or 4x).
        Set upscale=False only if you want to disable AI enhancement entirely.
        """
        """
        Enhance image using AI model.
        
        Args:
            image: Input image (BGR format from OpenCV)
            upscale: Whether to upscale the image
            
        Returns:
            Enhanced image and report
        """
        report = {
            'model_used': self.model_name,
            'operations': [],
            'improvements': {}
        }
        
        if self.model_name == 'traditional':
            # Fallback to traditional methods
            return image, report
        
        try:
            # Use portable executable if available (fastest and most reliable)
            if self.exe_path and os.path.exists(self.exe_path):
                import tempfile
                
                # Create temporary files
                with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_input:
                    input_path = tmp_input.name
                    cv2.imwrite(input_path, image)
                
                with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_output:
                    output_path = tmp_output.name
                
                try:
                    # Build command
                    # Note: The portable executable always applies the scale factor
                    # The upscale parameter here just controls if we use AI enhancement at all
                    if not upscale:
                        # If upscale is False, skip AI enhancement
                        raise ValueError("Upscaling disabled, skipping AI enhancement")
                    
                    cmd = [
                        self.exe_path,
                        '-i', input_path,
                        '-o', output_path,
                        '-n', self.model_name_exe,
                        '-s', str(self.model_scale),  # This always applies the scale (2x or 4x)
                        '-f', 'png'
                    ]
                    
                    # Run Real-ESRGAN
                    result = subprocess.run(
                        cmd,
                        capture_output=True,
                        text=True,
                        cwd=os.path.dirname(self.exe_path)
                    )
                    
                    if result.returncode != 0:
                        raise RuntimeError(f"Real-ESRGAN failed: {result.stderr}")
                    
                    # Load enhanced image
                    if not os.path.exists(output_path):
                        raise FileNotFoundError(f"Output file not created: {output_path}")
                    
                    enhanced = cv2.imread(output_path)
                    
                    if enhanced is None:
                        raise ValueError("Failed to load enhanced image")
                    
                    report['operations'].append('ai_enhancement')
                    report['operations'].append(f'{self.model_scale}x_upscaling' if upscale else 'ai_enhancement_only')
                    report['improvements']['ai_enhancement'] = f'Enhanced using Real-ESRGAN {self.model_scale}x (portable executable)'
                    
                    if upscale:
                        original_size = image.shape[:2]
                        new_size = enhanced.shape[:2]
                        report['improvements']['resolution'] = f'Upscaled from {original_size[1]}x{original_size[0]} to {new_size[1]}x{new_size[0]}'
                    
                    return enhanced, report
                    
                finally:
                    # Clean up temporary files
                    try:
                        if os.path.exists(input_path):
                            os.unlink(input_path)
                        if os.path.exists(output_path):
                            os.unlink(output_path)
                    except:
                        pass
            
            # Convert BGR to RGB for other AI models
            if len(image.shape) == 3:
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            
            # Use py-real-esrgan if available (simpler)
            if self.py_realesrgan_model is not None:
                from PIL import Image
                import torch
                
                # Convert numpy array to PIL Image
                pil_image = Image.fromarray(image_rgb)
                
                # Enhance using py-real-esrgan
                output_pil = self.py_realesrgan_model.predict(pil_image)
                
                # Convert back to numpy array
                output = np.array(output_pil)
                
                report['operations'].append('ai_enhancement')
                report['operations'].append(f'{self.model_scale}x_upscaling' if upscale else 'ai_enhancement_only')
                report['improvements']['ai_enhancement'] = f'Enhanced using Real-ESRGAN {self.model_scale}x model (py-real-esrgan)'
                
                if upscale:
                    original_size = image_rgb.shape[:2]
                    new_size = output.shape[:2]
                    report['improvements']['resolution'] = f'Upscaled from {original_size[1]}x{original_size[0]} to {new_size[1]}x{new_size[0]}'
                
                # Convert back to BGR
                if len(output.shape) == 3:
                    enhanced = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
                else:
                    enhanced = cv2.cvtColor(output, cv2.COLOR_GRAY2BGR)
                
                return enhanced, report
            
            # Use original realesrgan package if available
            elif self.model_name == 'realesrgan' and self.upsampler is not None:
                output, _ = self.upsampler.enhance(image_rgb, outscale=self.model_scale if upscale else 1)
                report['operations'].append('ai_enhancement')
                report['operations'].append(f'{self.model_scale}x_upscaling' if upscale else 'ai_enhancement_only')
                report['improvements']['ai_enhancement'] = f'Enhanced using Real-ESRGAN {self.model_scale}x model'
                
                if upscale:
                    original_size = image_rgb.shape[:2]
                    new_size = output.shape[:2]
                    report['improvements']['resolution'] = f'Upscaled from {original_size[1]}x{original_size[0]} to {new_size[1]}x{new_size[0]}'
                
                # Convert back to BGR
                if len(output.shape) == 3:
                    enhanced = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
                else:
                    enhanced = cv2.cvtColor(output, cv2.COLOR_GRAY2BGR)
                
                return enhanced, report
            
        except Exception as e:
            logger.error(f"AI enhancement failed: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            logger.info("Falling back to traditional methods")
            return image, report
        
        return image, report
    
    def is_available(self) -> bool:
        """Check if AI enhancement is available."""
        return (self.model_name != 'traditional' and 
                (self.upsampler is not None or 
                 self.py_realesrgan_model is not None or 
                 (self.exe_path is not None and os.path.exists(self.exe_path))))


def create_ai_enhancer(model_name: str = 'realesrgan', model_scale: int = 4, exe_path: Optional[str] = None) -> Optional[AIEnhancement]:
    """
    Create an AI enhancer instance.
    
    Args:
        model_name: Model to use ('realesrgan', 'esrgan', 'traditional')
        model_scale: Upscaling factor
        exe_path: Path to portable Real-ESRGAN executable (optional)
        
    Returns:
        AIEnhancement instance or None if not available
    """
    try:
        enhancer = AIEnhancement(model_name=model_name, model_scale=model_scale, exe_path=exe_path)
        if enhancer.is_available():
            return enhancer
        else:
            logger.warning("AI enhancement not available, using traditional methods")
            return None
    except Exception as e:
        logger.error(f"Failed to create AI enhancer: {e}")
        return None


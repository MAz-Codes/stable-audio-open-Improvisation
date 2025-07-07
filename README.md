# Fragment Generation for Live Improvisation

## Overview

Built on Stable Audio Open and stable-audio-tools, This tool generates audio fragments using the Stable Audio Open model with a fine-tuned checkpoint. It creates cyclic output files (output01.wav through output08.wav) for live performance use while archiving all generated audio for later reference.

## Features

- **Cyclic Fragment Generation**: Rotates through 8 output files (output01.wav - output08.wav) for seamless live performance integration. You can change this to a different number.
- **Automatic Archiving**: Saves all raw, unprocessed audio with sequential numbering
- **Audio Post-Processing**: Automatically removes silence and applies fade-in/fade-out
- **Gradio Web Interface**: Easy-to-use web interface for prompt-based generation
- **GPU Acceleration**: Optimized for CUDA-enabled devices with optional mixed precision

## Requirements

### Hardware
- NVIDIA GPU with CUDA support (recommended)
- Minimum 8GB VRAM for standard operation
- 16GB+ VRAM recommended

## Setup Instructions

### Install Dependencies
```bash
pip install torch torchaudio gradio einops
pip install soundfile librosa
pip install stable-audio-tools
```

### 1. Download Model Checkpoint

You need to obtain a Stable Audio Open checkpoint file:

**Option A: Official Stable Audio Open**
- Visit the [Stable Audio Open Hugging Face repository](https://huggingface.co/stabilityai/stable-audio-open-1.0)
- Download the model checkpoint file
- Place it in your project directory

**Option B: Fine-tuned Checkpoint**
- Use your own fine-tuned checkpoint
- Ensure it's compatible with the Stable Audio Open architecture

### 2. Configure Paths

Edit the configuration section in `fragment_generation.py`:

```python
# ===== CONFIGURATION SECTION =====
# Paths
LOCAL_CHECKPOINT_PATH = "your_checkpoint.ckpt"  # Your own or use the one in stable-audio-open-1.0
LOCAL_CONFIG_PATH = "./stable-audio-open-1.0/model_config.json"  # Provided
FRAGMENTS_DIR = "path/to/your/fragments"  # Choose your fragment directory
ARCHIVE_DIR = "path/to/your/fragments/archive"  # Archive subdirectory
```

### 3. Directory Structure Requirements

**Important**: For live performance integration with the provided Max for Live devices, the fragment directory must be the same folder where your Max for Live device is located.

```
your_project_folder/
├── fragment_generation.py
├── post_processing.py              # Provided - handles silence removal
├── stable-audio-open-1.0/
│   └── model_config.json           # Provided - model configuration
├── your_checkpoint_file.ckpt       # Your chosen checkpoint
└── your_fragments_folder/          # Your chosen fragment directory
    ├── output01.wav                # Generated fragments (cyclic)
    ├── output02.wav
    ├── ...
    ├── output08.wav
    └── archive/                    # Archive directory
        ├── output001.wav           # Sequential archive files
        ├── output002.wav
        └── ...
```

## Configuration Options

### Model Settings
```python
MODEL_HALF_PRECISION = False  # Set to True for float16 (faster, less VRAM)
```

### Audio Generation Parameters
```python
DEFAULT_STEPS = 100                    # Diffusion steps (50-120 recommended)
DEFAULT_CFG_SCALE = 7                  # Classifier-free guidance scale
DEFAULT_SIGMA_MIN = 0.3                # Minimum noise level
DEFAULT_SIGMA_MAX = 500                # Maximum noise level
DEFAULT_SAMPLER_TYPE = "dpmpp-3m-sde"  # Diffusion sampler
DEFAULT_SECONDS_TOTAL = 47             # Audio length in seconds
```

### Post-Processing Settings
```python
DEFAULT_SILENCE_THRESHOLD_DB = -40.0   # Silence detection threshold
DEFAULT_FADE_MS = 10.0                 # Fade duration in milliseconds
```

## Usage

### Running the Application

1. Ensure all paths are correctly configured
2. Run the script:
   ```bash
   python fragment_generation.py
   ```
3. Open the provided URL in your web browser
4. Enter your text prompt and click "Generate Audio"

### Understanding the Output

**Fragment Files**: 
- `output01.wav` through `output08.wav` in your fragments directory
- These files are overwritten cyclically for live performance use
- Post-processed with silence removal and fades

**Archive Files**:
- `output001.wav`, `output002.wav`, etc. in the archive subdirectory
- Sequential numbering, never overwritten
- Raw, unprocessed audio for reference

### Live Performance Integration

The cyclic fragment system is designed for Max for Live integration:
1. Set `FRAGMENTS_DIR` to the same folder as your Max for Live device
2. The corresponding Max device will import `output01.wav` through `output08.wav` upon trigger
3. Insert max devices on Ableton tracks (up to 8) and perform.

## Provided Files

### `post_processing.py`
- Handles automatic silence removal
- Applies fade-in and fade-out effects
- Maintains stereo compatibility
- **Do not modify** unless you understand the audio processing pipeline

### `model_config.json`
- Contains model architecture configuration
- Required for proper model instantiation
- **Do not modify** - specific to Stable Audio Open architecture

## Troubleshooting

### Common Issues

**CUDA Out of Memory**:
- Set `MODEL_HALF_PRECISION = True`
- Reduce `DEFAULT_STEPS`
- Close other GPU-intensive applications

**File Path Errors**:
- Use absolute paths for all directory configurations
- Ensure directories exist or are creatable
- Check file permissions

**Model Loading Errors**:
- Verify checkpoint file compatibility
- Ensure `model_config.json` matches your checkpoint
- Check CUDA installation for GPU usage. 
- For NVIDIA 5000 series, I used CUDA 12.8 alongside PyTorch Version: 2.7.0.

### Performance Optimization

**For Faster Generation**:
- Enable half precision: `MODEL_HALF_PRECISION = True`
- Reduce steps: `DEFAULT_STEPS = 50`
- Use GPU: Ensure CUDA is properly installed

**For Better Quality**:
- Increase steps: `DEFAULT_STEPS = 120`
- Use full precision: `MODEL_HALF_PRECISION = False`
- Experiment with `cfg_scale` values (6-12 range)

## Technical Notes

### Audio Specifications
- Sample Rate: Determined by model config 
- Bit Depth: 16-bit signed integer output
- Channels: Stereo (2 channels)
- Format: WAV

### Processing Pipeline
1. Text prompt → Model conditioning
2. Diffusion-based audio generation
3. Audio normalization and format conversion
4. Temporary file creation
5. Post-processing (silence removal, fades)
6. Cyclic fragment saving
7. Archive copy creation
8. Cleanup

## License and Attribution

Please respect the licenses of:
- Stable Audio Open model
- Any fine-tuned checkpoints you use
- The stable-audio-tools library

Ensure proper attribution when using generated audio in productions or performances.

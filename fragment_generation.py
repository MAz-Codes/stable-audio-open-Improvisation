import torch
import json
import os
import torchaudio
import time
import gradio as gr
from einops import rearrange
from stable_audio_tools.models.factory import create_model_from_config
from stable_audio_tools.models.utils import load_ckpt_state_dict
from stable_audio_tools.training.utils import copy_state_dict
from stable_audio_tools.inference.generation import generate_diffusion_cond

from post_processing import clean_audio_file_stereo

# ===== CONFIGURATION SECTION =====
# Paths
LOCAL_CHECKPOINT_PATH = "your_checkpoint_file.ckpt"  # Update path or one in stable-audio-open-1.0
LOCAL_CONFIG_PATH = "./stable-audio-open-1.0/model_config.json"  # Provided
FRAGMENTS_DIR = "path/to/your/fragments"  # Choose your fragment directory
ARCHIVE_DIR = "path/to/your/fragments/archive"  # Archive subdirectory

# Device configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Model configuration
MODEL_HALF_PRECISION = False  # Set to True to use float16 for speed/memory

# Audio generation defaults
DEFAULT_STEPS = 100
DEFAULT_CFG_SCALE = 7
DEFAULT_SIGMA_MIN = 0.3
DEFAULT_SIGMA_MAX = 500
DEFAULT_SAMPLER_TYPE = "dpmpp-3m-sde"
DEFAULT_SECONDS_TOTAL = 47
DEFAULT_SILENCE_THRESHOLD_DB = -40.0
DEFAULT_FADE_MS = 10.0


current_cycle_index = 1

def get_cyclic_output_filename(base_folder, base_name="output", extension=".wav"):
    """
    Rotate among output01.wav, output02.wav, ..., output08.wav, then wrap back to output01.wav.
    """
    global current_cycle_index
    filename = f"{base_name}{current_cycle_index:02d}{extension}"
    full_path = os.path.join(base_folder, filename)
    current_cycle_index = (current_cycle_index % 8) + 1
    return full_path

def get_next_archive_filename(folder, base_name="output", extension=".wav"):
    """
    Find the next unused filename in the archive folder (e.g., output001.wav, output002.wav, ...).
    """
    files = os.listdir(folder)
    index = 1
    while True:
        new_filename = f"{base_name}{index:03d}{extension}"
        if new_filename not in files:
            return os.path.join(folder, new_filename)
        index += 1

def load_model_local(model_config: dict, model_ckpt_path: str,
                     pretransform_ckpt_path: str = None,
                     device: str = "cuda", model_half: bool = False):
    """
    Load from a local config JSON + wrapped .ckpt file.
    """
    global model, sample_rate, sample_size

    if model_config is None or model_ckpt_path is None:
        raise ValueError("Must supply both model_config and model_ckpt_path for local inference.")

    # 1. Instantiate model from JSON config
    print("Creating model from config …")
    model = create_model_from_config(model_config)

    # 2. Load weights from local checkpoint
    print(f"Loading model checkpoint from {model_ckpt_path} …")
    state_dict = load_ckpt_state_dict(model_ckpt_path)
    copy_state_dict(model, state_dict)

    # 3. Extract sample_rate and sample_size from config
    sample_rate = model_config["sample_rate"]
    sample_size = model_config["sample_size"]

    # 4. (Optional) Load a separate "pretransform" checkpoint (if provided)
    if pretransform_ckpt_path is not None:
        print(f"Loading pretransform checkpoint from {pretransform_ckpt_path} …")
        pretransform_state_dict = load_ckpt_state_dict(pretransform_ckpt_path)
        model.pretransform.load_state_dict(pretransform_state_dict, strict=False)
        print("Done loading pretransform")

    # 5. Move to device, set eval mode
    model.to(device).eval().requires_grad_(False)

    # 6. (Optional) Cast to float16 if desired
    if model_half:
        model.to(torch.float16)

    print("Done loading model (local).")
    return model, model_config

# ===== MODEL LOADING AND INITIALIZATION =====
# Load JSON config
with open(LOCAL_CONFIG_PATH, 'r') as f:
    model_config = json.load(f)
print("Successfully loaded local model configuration.")

# Instantiate model using only local checkpoint
model, model_config = load_model_local(
    model_config=model_config,
    model_ckpt_path=LOCAL_CHECKPOINT_PATH,
    device=DEVICE,
    model_half=MODEL_HALF_PRECISION
)

# Compile to reduce Python overhead (PyTorch 2.0+)
if DEVICE.startswith("cuda"):
    model = torch.compile(model, mode="reduce-overhead")

sample_rate = model_config["sample_rate"]
sample_size = model_config["sample_size"]
print(f"Sample rate: {sample_rate}, Sample size: {sample_size}")
print("Model loaded successfully.")
# ===== END MODEL INITIALIZATION =====

@torch.no_grad()
def generate_audio(
    prompt: str,
    steps: int = DEFAULT_STEPS,
    cfg_scale: float = DEFAULT_CFG_SCALE,
    sigma_min: float = DEFAULT_SIGMA_MIN,
    sigma_max: float = DEFAULT_SIGMA_MAX,
    sampler_type: str = DEFAULT_SAMPLER_TYPE
):
    start_time = time.time()

    conditioning = [{
        "prompt": prompt,
        "seconds_start": 0,
        "seconds_total": DEFAULT_SECONDS_TOTAL
    }]

    # 1. Diffusion inference (mixed‑precision)
    with torch.inference_mode(), torch.cuda.amp.autocast():
        raw = generate_diffusion_cond(
            model,
            steps=steps,
            cfg_scale=cfg_scale,
            conditioning=conditioning,
            sample_size=sample_size,
            sigma_min=sigma_min,
            sigma_max=sigma_max,
            sampler_type=sampler_type,
            device=DEVICE
        )

    # 2. Reshape & normalize to float32 [-1,1]
    audio = rearrange(raw, "b d n -> d (b n)").to(torch.float32)
    audio = audio / audio.abs().max()
    audio_int16 = (audio.clamp(-1,1) * 32767).to(torch.int16).cpu()

    # Ensure output directories
    os.makedirs(FRAGMENTS_DIR, exist_ok=True)
    os.makedirs(ARCHIVE_DIR, exist_ok=True)

    # 3. Save raw to temp for trimming
    temp_raw = os.path.join(FRAGMENTS_DIR, "__temp_raw.wav")
    torchaudio.save(temp_raw, audio_int16, sample_rate)

    # 4. Clean & overwrite rotating fragment
    frag_path = get_cyclic_output_filename(FRAGMENTS_DIR)
    clean_audio_file_stereo(
        input_path=temp_raw,
        output_path=frag_path,
        sample_rate=sample_rate,
        silence_threshold_db=DEFAULT_SILENCE_THRESHOLD_DB,
        fade_ms=DEFAULT_FADE_MS
    )

    # 5. Archive the raw, untrimmed output
    archive_path = get_next_archive_filename(ARCHIVE_DIR)
    torchaudio.save(archive_path, audio_int16, sample_rate)

    # Remove the temporary raw file
    os.remove(temp_raw)

    # 6. Load cleaned fragment for Gradio
    wav, sr = torchaudio.load(frag_path)
    wav_f = wav.to(torch.float32).div(32767.0).t().numpy()

    inference_time = time.time() - start_time
    # Include both filenames in the log
    fragment_filename = os.path.basename(frag_path)
    archive_filename = os.path.basename(archive_path)
    return f"Inference time: {inference_time:.2f} seconds - Using local RTX 5080.\nFragment saved as: {fragment_filename}\nArchived as: {archive_filename}"

# ===== GRADIO INTERFACE =====
with gr.Blocks() as demo:
    gr.Markdown("##(un)Stable (dis)Connection Fragment Generation UI")
    prompt_input = gr.Textbox(label="Enter your prompt here:", lines=2)
    generate_button = gr.Button("Generate Audio")
    inference_time_output = gr.Textbox(label="Inference Log", interactive=False)

    generate_button.click(
        fn=generate_audio,
        inputs=prompt_input,
        outputs=[inference_time_output]
    )

demo.launch(debug=True, share=True, inline=True)

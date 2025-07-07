import numpy as np
import librosa
import soundfile as sf

"""
    I made this script for the paper "Live Improvisation with Fine-Tuned Generative AI".
    It removes silences and applies fades fo generated fragments:
    
    1) Load an audio file.
    2) Trim leading/trailing silence below `silence_threshold_db`.
       (Silence detection is based on the max amplitude across channels.)
    3) Apply a short fade-in/fade-out (`fade_ms` milliseconds) per channel.
    4) Save the result to `output_path`.

    In my pipeline, this is integrated into the generation process and is run after inference.
    Don't forget to change the paths to your own and also play with the thresholds.
"""


def clean_audio_file_stereo(
    input_path: "input_path",
    output_path: "output_path",
    sample_rate: int = 44100,
    silence_threshold_db: float = -40.0,
    fade_ms: float = 10.0
) -> None:

    # ------------------------------------------------
    # 1) Load audio with librosa
    # ------------------------------------------------
    audio, sr = librosa.load(input_path, sr=sample_rate, mono=False)
    if audio.ndim == 1:
        audio = np.expand_dims(audio, axis=0)
    channels, total_samples = audio.shape
    print(
        f"Loaded '{input_path}' with shape {audio.shape} (ch={channels}, samples={total_samples})")

    # ------------------------------------------------
    # 2) Trim silence based on overall max amplitude across channels
    # ------------------------------------------------
    amplitude_db = librosa.amplitude_to_db(np.abs(audio), ref=np.max)
    amplitude_db_max = np.max(amplitude_db, axis=0)
    non_silent = amplitude_db_max > silence_threshold_db

    if not np.any(non_silent):

        print("ATTENTION: Entire file is silent; disregarding the file.")
        return

    start_idx = np.where(non_silent)[0][0]
    end_idx = np.where(non_silent)[0][-1]
    trimmed = audio[:, start_idx:end_idx+1]
    trimmed_samples = trimmed.shape[1]
    print(
        f"Trimmed silence: start={start_idx}, end={end_idx}, new shape={trimmed.shape}")
    # ------------------------------------------------
    # 3) Apply fades
    # ------------------------------------------------
    fade_samples = int((fade_ms / 1000.0) * sr)
    if fade_samples > 0 and fade_samples * 2 < trimmed_samples:
        fade_in = np.linspace(0.0, 1.0, fade_samples)
        fade_out = np.linspace(1.0, 0.0, fade_samples)

        for c in range(channels):
            trimmed[c, :fade_samples] *= fade_in
            trimmed[c, -fade_samples:] *= fade_out

    # ------------------------------------------------
    # 4) Save the cleaned file
    # ------------------------------------------------
    sf.write(output_path, trimmed.T, sr)
    print(f"Saved cleaned audio to '{output_path}'")

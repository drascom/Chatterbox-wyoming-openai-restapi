# File: engine.py
# Core TTS model loading and speech generation logic.

import logging
import random
import numpy as np
import torch
from typing import Optional, Tuple
from pathlib import Path

from chatterbox.tts import ChatterboxTTS  # Main TTS engine class
from chatterbox.mtl_tts import ChatterboxMultilingualTTS
from chatterbox.models.s3gen.const import (
    S3GEN_SR,
)  # Default sample rate from the engine

# Import the singleton config_manager
from config import config_manager

logger = logging.getLogger(__name__)

# --- Global Module Variables ---
chatterbox_model: Optional[ChatterboxTTS] = None
MODEL_LOADED: bool = False
model_device: Optional[str] = (
    None  # Stores the resolved device string ('cuda' or 'cpu')
)
chatterbox_multilingual_model: Optional[ChatterboxMultilingualTTS] = None
MULTILINGUAL_MODEL_LOADED: bool = False
MIN_SAFE_TEMPERATURE = 0.1
MODEL_STATUS: dict = {"state": "idle", "detail": "Model load not started."}


def set_seed(seed_value: int):
    """
    Sets the seed for torch, random, and numpy for reproducibility.
    This is called if a non-zero seed is provided for generation.
    """
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)  # if using multi-GPU
    random.seed(seed_value)
    np.random.seed(seed_value)
    logger.info(f"Global seed set to: {seed_value}")


def _clamp_temperature(value: Optional[float]) -> float:
    """Clamp temperature to a safe minimum to avoid engine instability."""
    if value is None:
        return MIN_SAFE_TEMPERATURE
    if value < MIN_SAFE_TEMPERATURE:
        logger.warning(
            "Received temperature %.4f below safe minimum %.2f; clamping to minimum.",
            value,
            MIN_SAFE_TEMPERATURE,
        )
        return MIN_SAFE_TEMPERATURE
    return value


def load_model() -> bool:
    """
    Loads the TTS model.
    This version directly attempts to load from the Hugging Face repository (or its cache)
    using `from_pretrained`, bypassing the local `paths.model_cache` directory.
    Updates global variables `chatterbox_model`, `MODEL_LOADED`, and `model_device`.

    Returns:
        bool: True if the model was loaded successfully, False otherwise.
    """
    global chatterbox_model, MODEL_LOADED, model_device
    global chatterbox_multilingual_model, MULTILINGUAL_MODEL_LOADED
    MODEL_STATUS.update({"state": "loading", "detail": "Starting model load..."})

    enable_multilingual = config_manager.get_bool(
        "tts_engine.multilingual_model_enabled", False
    )

    if MODEL_LOADED:
        logger.info("TTS model is already loaded.")
        if enable_multilingual and not MULTILINGUAL_MODEL_LOADED and model_device:
            _load_multilingual_model(model_device)
        MODEL_STATUS.update({"state": "ready", "detail": "Model already loaded."})
        return True

    try:
        # Determine processing device
        device_setting = config_manager.get_string("tts_engine.device", "auto")
        if device_setting == "auto":
            resolved_device_str = "cuda" if torch.cuda.is_available() else "cpu"
        elif device_setting in ["cuda", "cpu"]:
            resolved_device_str = device_setting
        else:
            logger.warning(
                f"Invalid device setting '{device_setting}', defaulting to auto-detection."
            )
            resolved_device_str = "cuda" if torch.cuda.is_available() else "cpu"

        model_device = resolved_device_str
        logger.info(f"Attempting to load TTS model on device: {model_device}")

        # Get configured model_repo_id for logging and context,
        # though from_pretrained might use its own internal default if not overridden.
        model_repo_id_config = config_manager.get_string(
            "model.repo_id", "ResembleAI/chatterbox"
        )

        logger.info(
            f"Attempting to load model directly using from_pretrained (expected from Hugging Face repository: {model_repo_id_config} or library default)."
        )
        try:
            # Directly use from_pretrained. This will utilize the standard Hugging Face cache.
            # The ChatterboxTTS.from_pretrained method handles downloading if the model is not in the cache.
            chatterbox_model = ChatterboxTTS.from_pretrained(device=model_device)
            # The actual repo ID used by from_pretrained is often internal to the library,
            # but logging the configured one provides user context.
            logger.info(
                f"Successfully loaded TTS model using from_pretrained (expected from '{model_repo_id_config}' or library default)."
            )
        except Exception as e_hf:
            logger.error(
                f"Failed to load model using from_pretrained (expected from '{model_repo_id_config}' or library default): {e_hf}",
                exc_info=True,
            )
            MODEL_STATUS.update(
                {"state": "failed", "detail": f"Model load failed: {e_hf}"}
            )
            chatterbox_model = None
            MODEL_LOADED = False
            return False

        MODEL_LOADED = True
        if chatterbox_model:
            logger.info(
                f"TTS Model loaded successfully. Engine sample rate: {chatterbox_model.sr} Hz."
            )
            MODEL_STATUS.update(
                {
                    "state": "ready",
                    "detail": f"Model loaded (sample rate {chatterbox_model.sr} Hz).",
                }
            )
        else:
            logger.error(
                "Model loading sequence completed, but chatterbox_model is None. This indicates an unexpected issue."
            )
            MODEL_LOADED = False
            MODEL_STATUS.update(
                {
                    "state": "failed",
                    "detail": "Model object was None after load.",
                }
            )
            return False

        if enable_multilingual:
            _load_multilingual_model(model_device)
        else:
            # Ensure we don't hold onto a multilingual model when disabled.
            chatterbox_multilingual_model = None
            MULTILINGUAL_MODEL_LOADED = False
        return True

    except Exception as e:
        logger.error(
            f"An unexpected error occurred during model loading: {e}", exc_info=True
        )
        chatterbox_model = None
        MODEL_LOADED = False
        MODEL_STATUS.update(
            {"state": "failed", "detail": f"Unexpected error: {e}"}
        )
        return False


def _load_multilingual_model(device: str) -> bool:
    """
    Attempts to load the ChatterboxMultilingualTTS model onto the specified device.
    Returns True if successful; False otherwise.
    """
    global chatterbox_multilingual_model, MULTILINGUAL_MODEL_LOADED
    try:
        logger.info(f"Attempting to load multilingual TTS model on device: {device}")
        chatterbox_multilingual_model = ChatterboxMultilingualTTS.from_pretrained(device=device)
        MULTILINGUAL_MODEL_LOADED = True
        logger.info("Multilingual TTS model loaded successfully.")
        MODEL_STATUS.update(
            {
                "state": "ready",
                "detail": "Multilingual model loaded.",
            }
        )
        return True
    except Exception as e_multi:
        logger.error(
            f"Failed to load multilingual TTS model: {e_multi}", exc_info=True
        )
        chatterbox_multilingual_model = None
        MULTILINGUAL_MODEL_LOADED = False
        MODEL_STATUS.update(
            {
                "state": "failed",
                "detail": f"Multilingual model load failed: {e_multi}",
            }
        )
        return False


def synthesize(
    text: str,
    audio_prompt_path: Optional[str] = None,
    temperature: float = 0.8,
    exaggeration: float = 0.5,
    cfg_weight: float = 0.5,
    seed: int = 0,
    language: Optional[str] = None,
) -> Tuple[Optional[torch.Tensor], Optional[int]]:
    """
    Synthesizes audio from text using the loaded TTS model.

    Args:
        text: The text to synthesize.
        audio_prompt_path: Path to an audio file for voice cloning or predefined voice.
        temperature: Controls randomness in generation.
        exaggeration: Controls expressiveness.
        cfg_weight: Classifier-Free Guidance weight.
        seed: Random seed for generation. If 0, default randomness is used.
              If non-zero, a global seed is set for reproducibility.
        language: Optional language identifier passed to the multilingual TTS model.

    Returns:
        A tuple containing the audio waveform (torch.Tensor) and the sample rate (int),
        or (None, None) if synthesis fails.
    """
    global chatterbox_model

    if not MODEL_LOADED or chatterbox_model is None:
        logger.error("TTS model is not loaded. Cannot synthesize audio.")
        return None, None

    try:
        # Set seed globally if a specific seed value is provided and is non-zero.
        if seed != 0:
            logger.info(f"Applying user-provided seed for generation: {seed}")
            set_seed(seed)
        else:
            logger.info(
                "Using default (potentially random) generation behavior as seed is 0."
            )

        temperature_to_use = _clamp_temperature(temperature)

        logger.debug(
            f"Synthesizing with params: audio_prompt='{audio_prompt_path}', temp={temperature_to_use}, "
            f"exag={exaggeration}, cfg_weight={cfg_weight}, seed_applied_globally_if_nonzero={seed}, language={language}"
        )

        selected_model = chatterbox_model
        use_multilingual = (
            language is not None and chatterbox_multilingual_model is not None
        )
        if use_multilingual:
            logger.info(
                f"Using multilingual TTS model for language '{language}'."
            )
            selected_model = chatterbox_multilingual_model
        elif language is not None and chatterbox_multilingual_model is None:
            logger.warning(
                "Language '%s' requested but multilingual model is unavailable. Falling back to base TTS model.",
                language,
            )

        generate_kwargs = dict(
            text=text,
            audio_prompt_path=audio_prompt_path,
            temperature=temperature_to_use,
            exaggeration=exaggeration,
            cfg_weight=cfg_weight,
        )
        if use_multilingual and language:
            generate_kwargs["language_id"] = language

        # Call the selected model's generate method
        wav_tensor = selected_model.generate(**generate_kwargs)

        # The ChatterboxTTS.generate method already returns a CPU tensor.
        return wav_tensor, chatterbox_model.sr

    except Exception as e:
        logger.error(f"Error during TTS synthesis: {e}", exc_info=True)
        return None, None


def get_model_status() -> dict:
    """Return current model load status."""
    return MODEL_STATUS.copy()


def start_model_load_in_background():
    """Launch model loading in a background thread if not already ready or loading."""
    import threading

    if MODEL_STATUS.get("state") == "loading":
        return
    if MODEL_LOADED:
        MODEL_STATUS.update({"state": "ready", "detail": "Model already loaded."})
        return

    def _loader():
        try:
            load_model()
        except Exception as exc:
            logger.error(f"Background model load failed: {exc}", exc_info=True)
            MODEL_STATUS.update({"state": "failed", "detail": str(exc)})

    MODEL_STATUS.update({"state": "loading", "detail": "Downloading/loading model..."})
    thread = threading.Thread(target=_loader, name="model-loader", daemon=True)
    thread.start()


# --- End File: engine.py ---

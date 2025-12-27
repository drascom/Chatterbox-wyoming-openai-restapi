# File: wyoming_server.py
# Optional Wyoming protocol server for streaming PCM audio to Home Assistant.

import asyncio
import logging
import threading
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

import engine
import utils
from config import (
    config_manager,
    get_audio_sample_rate,
    get_gen_default_cfg_weight,
    get_gen_default_exaggeration,
    get_gen_default_language,
    get_gen_default_seed,
    get_gen_default_speed_factor,
    get_gen_default_temperature,
    get_predefined_voices_path,
    get_reference_audio_path,
    get_wyoming_advertise_name,
    get_wyoming_chunk_size,
    get_wyoming_channels,
    get_wyoming_enabled,
    get_wyoming_host,
    get_wyoming_languages,
    get_wyoming_pcm_width,
    get_wyoming_port,
    get_wyoming_sample_rate,
    get_wyoming_split_text,
)

from wyoming.audio import AudioChunk, AudioStart, AudioStop
from wyoming.info import Describe, Info, TtsProgram, TtsVoice, Attribution
from wyoming.server import AsyncServer
from wyoming.tts import Synthesize

logger = logging.getLogger(__name__)


class WyomingTTSService:
    """Handle Wyoming Describe/Synthesize events by reusing the existing engine pipeline."""

    def __init__(self):
        self.sample_rate = get_wyoming_sample_rate() or get_audio_sample_rate()
        self.channels = max(1, get_wyoming_channels())
        self.pcm_width = max(1, get_wyoming_pcm_width())
        self.split_text = get_wyoming_split_text()
        self.chunk_size = max(50, get_wyoming_chunk_size())
        self.languages = get_wyoming_languages() or [get_gen_default_language()]
        self.voice_catalog = utils.get_predefined_voices()
        self.reference_catalog = utils.get_valid_reference_files()

    async def handle_event(self, event, writer) -> None:
        """Dispatch Describe and Synthesize events."""
        try:
            if Describe.is_type(event.type):
                await self._handle_describe(writer)
            elif Synthesize.is_type(event.type):
                await self._handle_synthesize(event, writer)
            else:
                logger.warning(f"Unsupported Wyoming event received: {event.type}")
        except Exception as exc:
            logger.error(f"Error handling Wyoming event: {exc}", exc_info=True)
            try:
                await writer.write_event(AudioStop().event())
            except Exception:
                logger.debug("Failed to emit AudioStop after error.", exc_info=True)

    async def _handle_describe(self, writer) -> None:
        """Respond to Describe with available voices and capabilities."""
        voices: List[TtsVoice] = []
        for voice in self.voice_catalog:
            voices.append(
                TtsVoice(
                    name=voice["filename"],
                    description=voice.get("display_name", voice["filename"]),
                    languages=self.languages,
                    attribution=Attribution(name="Chatterbox", url="https://github.com/resemble-ai/chatterbox"),
                )
            )
        for reference_file in self.reference_catalog:
            voices.append(
                TtsVoice(
                    name=reference_file,
                    description=f"Reference audio {reference_file}",
                    languages=self.languages,
                    attribution=Attribution(name="Chatterbox", url="https://github.com/resemble-ai/chatterbox"),
                )
            )

        info = Info(
            tts=[
                TtsProgram(
                    name=get_wyoming_advertise_name(),
                    description="Chatterbox TTS over Wyoming",
                    attribution=Attribution(name="Chatterbox-TR-Api"),
                    installed=True,
                    voices=voices,
                )
            ]
        )
        await writer.write_event(info.event())
        logger.info("Responded to Wyoming Describe request.")

    async def _handle_synthesize(self, event, writer) -> None:
        """Generate audio for a Wyoming Synthesize request and stream PCM frames."""
        payload = Synthesize.from_event(event)
        text = payload.text or ""
        if not text.strip():
            logger.warning("Received empty text for Wyoming synthesis; sending stop.")
            await writer.write_event(AudioStop().event())
            return

        voice_name = payload.voice.name if payload.voice else None
        language = payload.language or get_gen_default_language()

        try:
            pcm_bytes, rate = await self._synthesize_to_pcm(
                text=text, voice_name=voice_name, language=language
            )
        except Exception as exc:
            logger.error(f"Failed Wyoming synthesis: {exc}", exc_info=True)
            await writer.write_event(AudioStop().event())
            return

        await writer.write_event(
            AudioStart(rate=rate, width=self.pcm_width, channels=self.channels).event()
        )
        chunk_size_bytes = 2048
        for start in range(0, len(pcm_bytes), chunk_size_bytes):
            await writer.write_event(
                AudioChunk(
                    rate=rate,
                    width=self.pcm_width,
                    channels=self.channels,
                    audio=pcm_bytes[start : start + chunk_size_bytes],
                ).event()
            )
        await writer.write_event(AudioStop().event())
        logger.info(
            "Completed Wyoming synthesis stream "
            f"(len={len(pcm_bytes)} bytes, rate={rate}, voice={voice_name or 'default'}, lang={language})."
        )

    async def _synthesize_to_pcm(
        self, text: str, voice_name: Optional[str], language: Optional[str]
    ) -> Tuple[bytes, int]:
        """Run the existing TTS pipeline and return PCM16 bytes and sample rate."""
        if not engine.MODEL_LOADED:
            raise RuntimeError("TTS engine model is not loaded.")

        voice_path = self._resolve_voice_path(voice_name)
        text_chunks = self._split_text_if_needed(text)

        audio_segments: List[np.ndarray] = []
        engine_sr: Optional[int] = None

        for chunk in text_chunks:
            tensor, sr = await asyncio.to_thread(
                engine.synthesize,
                chunk,
                str(voice_path) if voice_path else None,
                get_gen_default_temperature(),
                get_gen_default_exaggeration(),
                get_gen_default_cfg_weight(),
                get_gen_default_seed(),
                language,
            )
            if tensor is None or sr is None:
                raise RuntimeError("Engine returned no audio.")

            if engine_sr is None:
                engine_sr = sr
            speed_factor = get_gen_default_speed_factor()
            if speed_factor != 1.0:
                tensor, _ = utils.apply_speed_factor(tensor, sr, speed_factor)

            tensor_np = tensor.cpu().numpy()
            if tensor_np.ndim == 2:
                tensor_np = tensor_np.squeeze()

            if config_manager.get_bool("audio_processing.enable_silence_trimming", False):
                tensor_np = utils.trim_lead_trail_silence(tensor_np, sr)
            if config_manager.get_bool("audio_processing.enable_internal_silence_fix", False):
                tensor_np = utils.fix_internal_silence(tensor_np, sr)
            if (
                config_manager.get_bool("audio_processing.enable_unvoiced_removal", False)
                and utils.PARSELMOUTH_AVAILABLE
            ):
                tensor_np = utils.remove_long_unvoiced_segments(tensor_np, sr)

            audio_segments.append(tensor_np.astype(np.float32))

        if not audio_segments:
            raise RuntimeError("No audio segments were produced.")
        if engine_sr is None:
            raise RuntimeError("Engine sample rate could not be determined.")

        merged = audio_segments[0] if len(audio_segments) == 1 else np.concatenate(audio_segments)
        target_sr = self.sample_rate or engine_sr
        if target_sr != engine_sr and utils.LIBROSA_AVAILABLE:
            merged = utils.librosa.resample(  # type: ignore[attr-defined]
                y=merged, orig_sr=engine_sr, target_sr=target_sr
            )
            engine_sr = target_sr
        elif target_sr != engine_sr:
            logger.warning(
                "Librosa unavailable, streaming engine sample rate "
                f"{engine_sr}Hz instead of requested {target_sr}Hz."
            )

        pcm = np.clip(merged, -1.0, 1.0)
        pcm_int16 = (pcm * 32767).astype(np.int16)
        if self.channels > 1:
            pcm_int16 = np.repeat(pcm_int16[:, np.newaxis], self.channels, axis=1).astype(
                np.int16
            )
        return pcm_int16.tobytes(), engine_sr

    def _resolve_voice_path(self, voice_name: Optional[str]) -> Optional[Path]:
        """Return a path for the requested voice (predefined or reference)."""
        predefined_dir = get_predefined_voices_path(ensure_absolute=True)
        reference_dir = get_reference_audio_path(ensure_absolute=True)
        default_voice = config_manager.get_string("tts_engine.default_voice_id")

        candidates = [voice_name, default_voice]
        for candidate in candidates:
            if not candidate:
                continue
            candidate_path = predefined_dir / candidate
            if candidate_path.is_file():
                return candidate_path
            candidate_path = reference_dir / candidate
            if candidate_path.is_file():
                max_dur = config_manager.get_int(
                    "audio_output.max_reference_duration_sec", 30
                )
                is_valid, msg = utils.validate_reference_audio(candidate_path, max_dur)
                if is_valid:
                    return candidate_path
                logger.warning(f"Reference audio '{candidate}' invalid: {msg}")
        return None

    def _split_text_if_needed(self, text: str) -> List[str]:
        """Split long text into manageable chunks."""
        if not self.split_text:
            return [text]
        threshold = int(self.chunk_size * 1.5)
        if len(text) <= threshold:
            return [text]
        logger.info(f"Splitting text into chunks of size ~{self.chunk_size} for Wyoming synthesis.")
        return utils.chunk_text_by_sentences(text, self.chunk_size)


async def _run_wyoming_server() -> None:
    """Create and run the Wyoming TCP server."""
    host = get_wyoming_host()
    port = get_wyoming_port()
    handler = WyomingTTSService()
    server = AsyncServer.from_uri(f"tcp://{host}:{port}")
    logger.info(f"Starting Wyoming server on {host}:{port}")
    await server.run(handler.handle_event)


def start_wyoming_server_in_background() -> Optional[threading.Thread]:
    """Start the Wyoming server in a background thread if enabled."""
    if not get_wyoming_enabled():
        logger.info("Wyoming server not started (wyoming.enabled is False).")
        return None

    def runner():
        try:
            asyncio.run(_run_wyoming_server())
        except Exception as exc:
            logger.error(f"Wyoming server terminated unexpectedly: {exc}", exc_info=True)

    thread = threading.Thread(target=runner, name="wyoming-server", daemon=True)
    thread.start()
    return thread


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(_run_wyoming_server())

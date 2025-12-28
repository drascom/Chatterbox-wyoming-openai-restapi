# File: wyoming_server.py
# Optional Wyoming protocol server for streaming PCM audio to Home Assistant.

import asyncio
import logging
import threading
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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
    get_voice_language_map,
)

from wyoming.audio import AudioChunk, AudioStart, AudioStop
from wyoming.info import Describe, Info, TtsProgram, TtsVoice, Attribution
from wyoming.server import AsyncEventHandler, AsyncServer
from wyoming.tts import Synthesize

logger = logging.getLogger(__name__)


class WyomingTTSService(AsyncEventHandler):
    """Handle Wyoming Describe/Synthesize events by reusing the existing engine pipeline."""

    def __init__(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        super().__init__(reader, writer)
        self.sample_rate = get_wyoming_sample_rate() or get_audio_sample_rate()
        self.channels = max(1, get_wyoming_channels())
        self.pcm_width = max(1, get_wyoming_pcm_width())
        self.split_text = get_wyoming_split_text()
        self.chunk_size = max(50, get_wyoming_chunk_size())

        # --- LOGIC FIX START ---
        # 1. Get languages from config
        configured_langs = get_wyoming_languages() or []

        # 2. Get the engine default language (e.g., 'tr')
        engine_default = get_gen_default_language()

        # 3. If config is empty, use engine default
        if not configured_langs:
            configured_langs = [engine_default]

        # 4. Normalize (convert 'tr' to 'tr-TR')
        normalized_langs = self._normalize_languages(configured_langs)

        # 5. PRIORITY ENFORCEMENT:
        # If the engine is set to 'tr', ensure 'tr-TR' is at index 0
        if engine_default == 'tr' and 'tr-TR' in normalized_langs:
            normalized_langs.remove('tr-TR')
            normalized_langs.insert(0, 'tr-TR')
        elif engine_default == 'tr' and 'tr-TR' not in normalized_langs:
            normalized_langs.insert(0, 'tr-TR')

        self.languages = normalized_langs
        # --- LOGIC FIX END ---

        self.voice_catalog = utils.get_predefined_voices()
        self.reference_catalog = utils.get_valid_reference_files()
        self.voice_language_map = get_voice_language_map()
        self.voice_name_map = self._build_voice_name_map()

    @staticmethod
    def _format_language_label(language: Optional[str]) -> Optional[str]:
        """Format a short language label for UI/HA display (e.g., en -> EN)."""
        if not language:
            return None
        base = language.split("-")[0].strip()
        return base.upper() if base else None

    @staticmethod
    def _strip_language_label(voice_name: str) -> str:
        """Strip a trailing language label like ' (EN)' if present."""
        if voice_name.endswith(")") and " (" in voice_name:
            return voice_name.rsplit(" (", 1)[0]
        return voice_name

    def _build_voice_name_map(self) -> Dict[str, str]:
        """Build a lookup of HA-visible voice names to actual filenames."""
        name_map: Dict[str, str] = {}
        for voice in self.voice_catalog:
            filename = voice["filename"]
            display_name = voice.get("display_name", filename)
            name_map[filename] = filename
            name_map[display_name] = filename
            label = self._format_language_label(
                self.voice_language_map.get(filename)
            )
            if label:
                name_map[f"{display_name} ({label})"] = filename

        for reference_file in self.reference_catalog:
            display_name = Path(reference_file).stem.replace("_", " ").replace("-", " ")
            name_map[reference_file] = reference_file
            name_map[display_name] = reference_file
            label = self._format_language_label(
                self.voice_language_map.get(reference_file)
            )
            if label:
                name_map[f"{display_name} ({label})"] = reference_file

        return name_map

    def _resolve_voice_filename(self, voice_name: Optional[str]) -> Optional[str]:
        """Resolve a Wyoming/HA voice name to the actual filename."""
        if not voice_name:
            return None
        if voice_name in self.voice_name_map:
            return self.voice_name_map[voice_name]
        stripped = self._strip_language_label(voice_name)
        if stripped in self.voice_name_map:
            return self.voice_name_map[stripped]
        return voice_name

    @staticmethod
    def _normalize_languages(language_codes: List[str]) -> List[str]:
        """Normalize language codes to BCP 47 tags."""
        # Maps generic codes to the specific region HA expects
        fallback_regions = {
            "en": "en-GB",
            "tr": "tr-TR"
        }

        normalized = []
        for code in language_codes:
            if not code:
                continue

            # Clean string
            clean_code = code.strip()
            lower = clean_code.lower()

            # If it's already regional (e.g. tr-TR), keep it
            if "-" in clean_code:
                normalized.append(clean_code)
            # If it's short (e.g. tr), map it
            elif lower in fallback_regions:
                normalized.append(fallback_regions[lower])
            else:
                normalized.append(clean_code)

        # Deduplicate while preserving order
        seen = set()
        deduped = []
        for code in normalized:
            if code not in seen:
                seen.add(code)
                deduped.append(code)
        return deduped

    @staticmethod
    def _to_engine_language(language: Optional[str]) -> Optional[str]:
        """Convert BCP 47 tag to base language for the engine (e.g., en-US -> en)."""
        if not language:
            return language
        if "-" in language:
            return language.split("-")[0]
        return language

    async def handle_event(self, event) -> bool:
        """Dispatch Describe and Synthesize events."""
        try:
            if Describe.is_type(event.type):
                await self._handle_describe()
            elif Synthesize.is_type(event.type):
                await self._handle_synthesize(event)
            else:
                logger.warning(f"Unsupported Wyoming event received: {event.type}")
        except Exception as exc:
            logger.error(f"Error handling Wyoming event: {exc}", exc_info=True)
            try:
                await self.write_event(AudioStop().event())
            except Exception:
                logger.debug("Failed to emit AudioStop after error.", exc_info=True)
        return True

    async def _handle_describe(self) -> None:
        """Respond to Describe with available voices and capabilities."""
        self.voice_language_map = get_voice_language_map()
        self.voice_name_map = self._build_voice_name_map()
        voices: List[TtsVoice] = []
        for voice in self.voice_catalog:
            filename = voice["filename"]
            display_name = voice.get("display_name", filename)
            label = self._format_language_label(
                self.voice_language_map.get(filename)
            )
            friendly_name = (
                f"{display_name} ({label})" if label else display_name
            )
            voices.append(
                TtsVoice(
                    name=friendly_name,
                    description=friendly_name,
                    languages=self.languages,
                    attribution=Attribution(name="Chatterbox", url="https://github.com/resemble-ai/chatterbox"),
                    installed=True,
                    version=None,
                )
            )
        for reference_file in self.reference_catalog:
            display_name = Path(reference_file).stem.replace("_", " ").replace("-", " ")
            label = self._format_language_label(
                self.voice_language_map.get(reference_file)
            )
            friendly_name = (
                f"{display_name} ({label})" if label else display_name
            )
            voices.append(
                TtsVoice(
                    name=friendly_name,
                    description=friendly_name,
                    languages=self.languages,
                    attribution=Attribution(name="Chatterbox", url="https://github.com/resemble-ai/chatterbox"),
                    installed=True,
                    version=None,
                )
            )

        info = Info(
            tts=[
                TtsProgram(
                    name=get_wyoming_advertise_name(),
                    description="Chatterbox TTS over Wyoming",
                    attribution=Attribution(
                        name="Chatterbox-TR-Api",
                        url="https://github.com/resemble-ai/chatterbox",
                    ),
                    installed=True,
                    version=None,
                    voices=voices,
                )
            ]
        )
        await self.write_event(info.event())
        logger.info("Responded to Wyoming Describe request.")

    async def _handle_synthesize(self, event) -> None:
        """Generate audio for a Wyoming Synthesize request and stream PCM frames."""
        self.voice_language_map = get_voice_language_map()
        self.voice_name_map = self._build_voice_name_map()
        raw_event = getattr(event, "data", None)
        logger.info("Wyoming synth request received: %s", raw_event)
        payload = Synthesize.from_event(event)
        text = payload.text or ""
        if not text.strip():
            logger.warning("Received empty text for Wyoming synthesis; sending stop.")
            await self.write_event(AudioStop().event())
            return

        voice_name = payload.voice.name if payload.voice else None
        voice_filename = self._resolve_voice_filename(voice_name)
        requested_language = payload.voice.language if payload.voice and hasattr(payload.voice, "language") else None
        # Prefer requested voice language if provided; otherwise fall back to the first advertised language.
        language = get_gen_default_language()
        if requested_language:
            language = requested_language
        elif voice_filename:
            mapped_language = self.voice_language_map.get(voice_filename)
            if mapped_language:
                language = mapped_language
        elif self.languages:
            language = self.languages[0]

        # If no language requested, try to infer from voice filename suffix `_tr` or `_en`.
        if not requested_language and voice_filename:
            lower_name = voice_filename.lower()
            if lower_name.endswith("_tr.wav") or lower_name.endswith("_tr.mp3"):
                language = "tr-TR"
            elif lower_name.endswith("_en.wav") or lower_name.endswith("_en.mp3"):
                language = "en-US"

        logger.info(
            "Wyoming synth resolved: voice=%s, requested_lang=%s, resolved_lang=%s, text_len=%d",
            voice_name or "default",
            requested_language or "none",
            language,
            len(text),
        )

        try:
            pcm_bytes, rate = await self._synthesize_to_pcm(
                text=text, voice_name=voice_filename or voice_name, language=language
            )
        except Exception as exc:
            logger.error(f"Failed Wyoming synthesis: {exc}", exc_info=True)
            await self.write_event(AudioStop().event())
            return

        await self.write_event(
            AudioStart(rate=rate, width=self.pcm_width, channels=self.channels).event()
        )
        chunk_size_bytes = 2048
        for start in range(0, len(pcm_bytes), chunk_size_bytes):
            await self.write_event(
                AudioChunk(
                    rate=rate,
                    width=self.pcm_width,
                    channels=self.channels,
                    audio=pcm_bytes[start : start + chunk_size_bytes],
                ).event()
            )
        await self.write_event(AudioStop().event())
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
            language_for_engine = self._to_engine_language(language)
            tensor, sr = await asyncio.to_thread(
                engine.synthesize,
                chunk,
                str(voice_path) if voice_path else None,
                get_gen_default_temperature(),
                get_gen_default_exaggeration(),
                get_gen_default_cfg_weight(),
                get_gen_default_seed(),
                language_for_engine,
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
        voice_name = self._resolve_voice_filename(voice_name)
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
    server = AsyncServer.from_uri(f"tcp://{host}:{port}")
    logger.info(f"Starting Wyoming server on {host}:{port}")
    await server.run(WyomingTTSService)


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

import asyncio
import io
import json
import logging
import os
import time
import wave
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import requests

from wyoming.asr import Transcribe, Transcript
from wyoming.audio import AudioChunk, AudioStart, AudioStop
from wyoming.info import AsrModel, AsrProgram, Attribution, Describe, Info, TtsProgram, TtsVoice
from wyoming.server import AsyncEventHandler, AsyncServer
from wyoming.tts import Synthesize


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


TTS_UPSTREAM_URL = os.getenv("TTS_UPSTREAM_URL", "http://chatterbox-tts:8000").rstrip("/")
STT_UPSTREAM_URL = os.getenv("STT_UPSTREAM_URL", "http://whisper-stt:10400").rstrip("/")

WYOMING_TTS_HOST = os.getenv("WYOMING_TTS_HOST", "0.0.0.0")
WYOMING_TTS_PORT = int(os.getenv("WYOMING_TTS_PORT", "10200"))
WYOMING_STT_HOST = os.getenv("WYOMING_STT_HOST", "0.0.0.0")
WYOMING_STT_PORT = int(os.getenv("WYOMING_STT_PORT", "10300"))

WYOMING_ENABLE_TTS = os.getenv("WYOMING_ENABLE_TTS", "true").lower() == "true"
WYOMING_ENABLE_STT = os.getenv("WYOMING_ENABLE_STT", "true").lower() == "true"

WYOMING_ADVERTISE_NAME = os.getenv("WYOMING_ADVERTISE_NAME", "Chatterbox TTS (Wyoming)")
DEFAULT_TTS_MODEL = os.getenv("TTS_MODEL", "chatterbox")

VOICE_DIR = Path(os.getenv("VOICE_DIR", "voices"))
REFERENCE_DIR = Path(os.getenv("REFERENCE_DIR", "reference_audio"))
LANGUAGE_STORE_DIR = Path(os.getenv("LANGUAGE_STORE_DIR", "logs"))
LANGUAGE_TTL_SEC = int(os.getenv("LANGUAGE_TTL_SEC", "300"))

TTS_TIMEOUT = (3, 60)
STT_TIMEOUT = (3, 120)

HARD_CODED_TTS_LANGUAGES = ["tr-TR", "en-GB"]
STT_LANGUAGES = ["tr", "tr-TR"]


def _safe_client_key(client_key: str) -> str:
    return "".join(c if c.isalnum() or c in (".", "-", "_") else "_" for c in client_key)


def _language_file_path(client_key: str) -> Path:
    LANGUAGE_STORE_DIR.mkdir(parents=True, exist_ok=True)
    return LANGUAGE_STORE_DIR / f"wyoming_lang_{_safe_client_key(client_key)}.json"


def set_last_requested_language(client_key: str, language: str) -> None:
    payload = {"language": language, "ts": time.time()}
    path = _language_file_path(client_key)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle)
    _prune_stale_languages(payload["ts"])


def get_last_requested_language_for_client(client_key: str) -> Optional[str]:
    path = _language_file_path(client_key)
    if not path.is_file():
        return None

    try:
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        ts = payload.get("ts")
        if not isinstance(ts, (int, float)) or time.time() - ts > LANGUAGE_TTL_SEC:
            path.unlink(missing_ok=True)
            return None
        return payload.get("language")
    except Exception:
        return None


def _prune_stale_languages(now: float) -> None:
    if not LANGUAGE_STORE_DIR.is_dir():
        return
    for path in LANGUAGE_STORE_DIR.glob("wyoming_lang_*.json"):
        try:
            with open(path, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
            ts = payload.get("ts")
        except Exception:
            ts = None

        if not isinstance(ts, (int, float)) or now - ts > LANGUAGE_TTL_SEC:
            path.unlink(missing_ok=True)


def _client_key_from_writer(writer: asyncio.StreamWriter) -> str:
    peername = writer.get_extra_info("peername")
    if isinstance(peername, tuple) and len(peername) >= 2:
        return str(peername[0])
    return str(peername or "unknown")


def _format_language_label(language: Optional[str]) -> Optional[str]:
    if not language:
        return None
    base = language.split("-")[0].strip()
    return base.upper() if base else None


def _guess_language_for_voice(filename: str) -> Optional[str]:
    lower_name = filename.lower()
    if lower_name.endswith("_tr.wav") or lower_name.endswith("_tr.mp3"):
        return "tr-TR"
    if lower_name.endswith("_en.wav") or lower_name.endswith("_en.mp3"):
        return "en-GB"
    return None


def _scan_voice_files() -> List[str]:
    files: List[str] = []
    for directory in (VOICE_DIR, REFERENCE_DIR):
        if not directory.is_dir():
            continue
        for path in sorted(directory.iterdir()):
            if path.is_file() and path.suffix.lower() in (".wav", ".mp3"):
                files.append(path.name)
    seen = set()
    deduped: List[str] = []
    for name in files:
        if name not in seen:
            seen.add(name)
            deduped.append(name)
    return deduped


def _voice_name_map(voice_files: List[str]) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    for filename in voice_files:
        stem_name = Path(filename).stem.replace("_", " ").replace("-", " ")
        mapping[filename] = filename
        mapping[stem_name] = filename
        label = _format_language_label(_guess_language_for_voice(filename))
        if label:
            mapping[f"{stem_name} ({label})"] = filename
            mapping[f"{filename} ({label})"] = filename
    return mapping


def _strip_language_suffix(name: str) -> str:
    if name.endswith(")") and " (" in name:
        return name.rsplit(" (", 1)[0]
    return name


def _wav_bytes_to_pcm_int16(wav_bytes: bytes) -> tuple[bytes, int, int]:
    with wave.open(io.BytesIO(wav_bytes), "rb") as wav_file:
        channels = wav_file.getnchannels()
        sample_width = wav_file.getsampwidth()
        sample_rate = wav_file.getframerate()
        frames = wav_file.readframes(wav_file.getnframes())

    if sample_width == 2:
        pcm = np.frombuffer(frames, dtype=np.int16)
    elif sample_width == 1:
        pcm_u8 = np.frombuffer(frames, dtype=np.uint8).astype(np.int16)
        pcm = (pcm_u8 - 128) << 8
    elif sample_width == 4:
        pcm_i32 = np.frombuffer(frames, dtype=np.int32)
        pcm = (pcm_i32 >> 16).astype(np.int16)
    else:
        raise RuntimeError(f"Unsupported WAV sample width from TTS upstream: {sample_width}")

    return pcm.tobytes(), sample_rate, channels


class WyomingTTSProxyService(AsyncEventHandler):
    def __init__(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        super().__init__(reader, writer)
        self.voice_files = _scan_voice_files()
        self.name_map = _voice_name_map(self.voice_files)

    async def handle_event(self, event) -> bool:
        try:
            if Describe.is_type(event.type):
                await self._handle_describe()
            elif Synthesize.is_type(event.type):
                await self._handle_synthesize(event)
            else:
                logger.warning("Unsupported Wyoming TTS event: %s", event.type)
        except Exception as exc:
            logger.error("Wyoming TTS handler error: %s", exc, exc_info=True)
            try:
                await self.write_event(AudioStop().event())
            except Exception:
                pass
        return True

    async def _handle_describe(self) -> None:
        self.voice_files = _scan_voice_files()
        self.name_map = _voice_name_map(self.voice_files)

        voices: List[TtsVoice] = []
        for filename in self.voice_files:
            display_name = Path(filename).stem.replace("_", " ").replace("-", " ")
            label = _format_language_label(_guess_language_for_voice(filename))
            friendly_name = f"{display_name} ({label})" if label else display_name
            voices.append(
                TtsVoice(
                    name=friendly_name,
                    description=friendly_name,
                    languages=HARD_CODED_TTS_LANGUAGES,
                    attribution=Attribution(name="Chatterbox", url="https://github.com/resemble-ai/chatterbox"),
                    installed=True,
                    version=1.0,
                )
            )

        info = Info(
            tts=[
                TtsProgram(
                    name=WYOMING_ADVERTISE_NAME,
                    description="Chatterbox TTS over Wyoming (gateway)",
                    attribution=Attribution(name="Chatterbox-TR-Api", url="https://github.com/resemble-ai/chatterbox"),
                    installed=True,
                    version=None,
                    voices=voices,
                )
            ]
        )
        await self.write_event(info.event())

    async def _handle_synthesize(self, event) -> None:
        payload = Synthesize.from_event(event)
        text = (payload.text or "").strip()
        if not text:
            await self.write_event(AudioStop().event())
            return

        requested_voice = payload.voice.name if payload.voice else None
        voice_filename = self._resolve_voice_filename(requested_voice)
        if not voice_filename and self.voice_files:
            voice_filename = self.voice_files[0]

        try:
            wav_data = await asyncio.to_thread(self._call_tts_upstream, text, voice_filename)
            pcm_data, rate, channels = _wav_bytes_to_pcm_int16(wav_data)
        except Exception as exc:
            logger.error("TTS upstream call failed: %s", exc, exc_info=True)
            await self.write_event(AudioStop().event())
            return

        await self.write_event(AudioStart(rate=rate, width=2, channels=channels).event())
        chunk_size = 2048
        for start in range(0, len(pcm_data), chunk_size):
            await self.write_event(
                AudioChunk(
                    rate=rate,
                    width=2,
                    channels=channels,
                    audio=pcm_data[start : start + chunk_size],
                ).event()
            )
        await self.write_event(AudioStop().event())

    def _resolve_voice_filename(self, voice_name: Optional[str]) -> Optional[str]:
        if not voice_name:
            return None
        if voice_name in self.name_map:
            return self.name_map[voice_name]
        stripped = _strip_language_suffix(voice_name)
        if stripped in self.name_map:
            return self.name_map[stripped]
        return voice_name

    @staticmethod
    def _call_tts_upstream(text: str, voice_filename: Optional[str]) -> bytes:
        if not voice_filename:
            raise RuntimeError("No voice available for synthesis")

        response = requests.post(
            f"{TTS_UPSTREAM_URL}/v1/audio/speech",
            json={
                "model": DEFAULT_TTS_MODEL,
                "input": text,
                "voice": voice_filename,
                "response_format": "wav",
                "speed": 1.0,
            },
            timeout=TTS_TIMEOUT,
        )
        if response.status_code >= 400:
            raise RuntimeError(
                f"TTS upstream returned HTTP {response.status_code}: {response.text[:300]}"
            )
        return response.content


class WyomingSTTProxyService(AsyncEventHandler):
    def __init__(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        super().__init__(reader, writer)
        self.audio_buffer = io.BytesIO()
        self.is_receiving = False
        self.sample_rate = 16000
        self.channels = 1
        self.width = 2

    async def handle_event(self, event) -> bool:
        try:
            if AudioStart.is_type(event.type):
                start = AudioStart.from_event(event)
                self.sample_rate = start.rate
                self.channels = start.channels
                self.width = start.width
                self.audio_buffer = io.BytesIO()
                self.is_receiving = True
            elif AudioChunk.is_type(event.type):
                if self.is_receiving:
                    chunk = AudioChunk.from_event(event)
                    self.audio_buffer.write(chunk.audio)
            elif AudioStop.is_type(event.type):
                self.is_receiving = False
                await self._transcribe_and_respond()
            elif Transcribe.is_type(event.type):
                event_data = getattr(event, "data", None)
                if isinstance(event_data, dict):
                    language = event_data.get("language")
                    if language:
                        set_last_requested_language(self._client_key(), str(language))
            elif Describe.is_type(event.type):
                await self._handle_describe()
            else:
                logger.warning("Unsupported Wyoming STT event: %s", event.type)
        except Exception as exc:
            logger.error("Wyoming STT handler error: %s", exc, exc_info=True)
        return True

    async def _handle_describe(self) -> None:
        info = Info(
            asr=[
                AsrProgram(
                    name="whisper_tr",
                    description="Whisper Turkish STT (gateway)",
                    attribution=Attribution(name="OpenAI", url="https://github.com/openai/whisper"),
                    installed=True,
                    version="1.0.0",
                    models=[
                        AsrModel(
                            name="selimc/whisper-large-v3-turbo-turkish",
                            description="Whisper Turkish STT",
                            attribution=Attribution(name="OpenAI", url="https://github.com/openai/whisper"),
                            installed=True,
                            languages=STT_LANGUAGES,
                            version="1.0.0",
                        )
                    ],
                )
            ]
        )
        await self.write_event(info.event())

    async def _transcribe_and_respond(self) -> None:
        data = self.audio_buffer.getvalue()
        if not data:
            logger.warning("No audio received for STT transcription")
            return

        language = get_last_requested_language_for_client(self._client_key())

        try:
            text = await asyncio.to_thread(
                self._call_stt_upstream,
                data,
                self.sample_rate,
                self.channels,
                self.width,
                language,
            )
            if text:
                await self.write_event(Transcript(text=text).event())
        except Exception as exc:
            logger.error("STT upstream call failed: %s", exc, exc_info=True)

    @staticmethod
    def _call_stt_upstream(
        pcm_bytes: bytes,
        sample_rate: int,
        channels: int,
        width: int,
        language: Optional[str],
    ) -> str:
        params = {
            "sample_rate": sample_rate,
            "channels": channels,
            "pcm_width": width,
        }
        if language:
            params["language"] = language

        response = requests.post(
            f"{STT_UPSTREAM_URL}/internal/stt/transcribe",
            params=params,
            data=pcm_bytes,
            headers={"Content-Type": "application/octet-stream"},
            timeout=STT_TIMEOUT,
        )
        if response.status_code >= 400:
            raise RuntimeError(
                f"STT upstream returned HTTP {response.status_code}: {response.text[:300]}"
            )

        payload = response.json()
        return str(payload.get("text") or "").strip()

    def _client_key(self) -> str:
        return _client_key_from_writer(self.writer)


async def _run_tts_server() -> None:
    server = AsyncServer.from_uri(f"tcp://{WYOMING_TTS_HOST}:{WYOMING_TTS_PORT}")
    logger.info("Starting Wyoming TTS gateway on %s:%s", WYOMING_TTS_HOST, WYOMING_TTS_PORT)
    await server.run(WyomingTTSProxyService)


async def _run_stt_server() -> None:
    server = AsyncServer.from_uri(f"tcp://{WYOMING_STT_HOST}:{WYOMING_STT_PORT}")
    logger.info("Starting Wyoming STT gateway on %s:%s", WYOMING_STT_HOST, WYOMING_STT_PORT)
    await server.run(WyomingSTTProxyService)


async def main() -> None:
    tasks = []
    if WYOMING_ENABLE_TTS:
        tasks.append(asyncio.create_task(_run_tts_server()))
    if WYOMING_ENABLE_STT:
        tasks.append(asyncio.create_task(_run_stt_server()))

    if not tasks:
        logger.warning("Both WYOMING_ENABLE_TTS and WYOMING_ENABLE_STT are false. Exiting.")
        return

    await asyncio.gather(*tasks)


if __name__ == "__main__":
    asyncio.run(main())

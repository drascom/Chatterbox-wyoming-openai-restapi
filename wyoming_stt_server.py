import asyncio
import io
import logging
import json
import os
import threading
import time
import warnings

import numpy as np
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

from config import get_log_file_path, get_wyoming_stt_host, get_wyoming_stt_port
from wyoming.info import Describe, Info, AsrProgram, AsrModel, Attribution
from wyoming.server import AsyncEventHandler, AsyncServer
from wyoming.audio import AudioChunk, AudioStart, AudioStop
from wyoming.asr import Transcribe, Transcript

# CONFIGURATION
# ---------------------
MODEL_ID = "selimc/whisper-large-v3-turbo-turkish"
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
TORCH_DTYPE = torch.float16 if torch.cuda.is_available() else torch.float32
LANGUAGES = ["tr", "tr-TR"]
# ---------------------

logging.basicConfig(level=logging.INFO)
_LOGGER = logging.getLogger(__name__)
warnings.filterwarnings(
    "ignore",
    message=".*LoRACompatibleLinear.*",
    category=FutureWarning,
)

# Global model/processor
asr_model = None
asr_processor = None
_LANGUAGE_TTL_SEC = 300


def load_model():
    """Load the Whisper model once at startup."""
    global asr_model, asr_processor
    _LOGGER.info("Loading Whisper model '%s' on %s...", MODEL_ID, DEVICE)
    asr_model = AutoModelForSpeechSeq2Seq.from_pretrained(
        MODEL_ID, torch_dtype=TORCH_DTYPE, low_cpu_mem_usage=True, use_safetensors=True
    )
    asr_model.to(DEVICE)
    asr_model.eval()
    asr_processor = AutoProcessor.from_pretrained(MODEL_ID)
    _LOGGER.info("Whisper model loaded.")


class WyomingSTTHandler(AsyncEventHandler):
    """Event handler for Wyoming STT."""

    def __init__(self, reader, writer):
        super().__init__(reader, writer)
        self.audio_buffer = io.BytesIO()
        self.is_receiving = False

    async def handle_event(self, event) -> bool:
        if AudioStart.is_type(event.type):
            self.audio_buffer = io.BytesIO()
            self.is_receiving = True

        elif AudioChunk.is_type(event.type):
            if self.is_receiving:
                chunk = AudioChunk.from_event(event)
                self.audio_buffer.write(chunk.audio)

        elif AudioStop.is_type(event.type):
            self.is_receiving = False
            await self._transcribe()

        elif Transcribe.is_type(event.type):
            _LOGGER.info("Wyoming STT request: %s", self._event_summary(event))
            language = None
            event_data = getattr(event, "data", None)
            if isinstance(event_data, dict):
                language = event_data.get("language")
            if language:
                set_last_requested_language(self._client_key(), str(language))

        elif Describe.is_type(event.type):
            await self._handle_describe()

        return True

    @staticmethod
    def _event_summary(event) -> str:
        """Best-effort summary of a Wyoming event for logging."""
        event_data = getattr(event, "data", None)
        if event_data is None:
            return f"type={getattr(event, 'type', 'unknown')}"
        return f"type={getattr(event, 'type', 'unknown')}, data={event_data}"

    def _client_key(self) -> str:
        """Build a stable client key from the connection."""
        peername = self.writer.get_extra_info("peername")
        if isinstance(peername, tuple) and len(peername) >= 2:
            return f"{peername[0]}"
        return str(peername or "unknown")

    async def _handle_describe(self):
        """Tell Home Assistant we are a Turkish STT service."""
        info = Info(
            asr=[
                AsrProgram(
                    name="whisper_tr",
                    description="Whisper Turkish STT",
                    attribution=Attribution(name="OpenAI", url="https://github.com/openai/whisper"),
                    installed=True,
                    version="1.0.0",
                    models=[
                        AsrModel(
                            name=MODEL_ID,
                            description=f"Whisper {MODEL_ID} (Turkish)",
                            attribution=Attribution(name="OpenAI", url="https://github.com/openai/whisper"),
                            installed=True,
                            languages=LANGUAGES,
                            version="1.0.0",
                        )
                    ],
                )
            ]
        )
        await self.write_event(info.event())

    async def _transcribe(self):
        """Process the buffered audio and send text back."""
        global asr_model, asr_processor

        data = self.audio_buffer.getvalue()
        if not data:
            _LOGGER.warning("No audio data received.")
            return

        # Home Assistant sends 16kHz, 16-bit mono PCM by default
        audio_np = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0

        if asr_model is None or asr_processor is None:
            _LOGGER.error("ASR model not loaded.")
            return

        try:
            def run_inference():
                inputs = asr_processor(
                    audio_np,
                    sampling_rate=16000,
                    return_tensors="pt",
                    return_attention_mask=True,
                )
                input_features = inputs.input_features.to(
                    DEVICE, dtype=asr_model.dtype
                )
                attention_mask = None
                if hasattr(inputs, "attention_mask") and inputs.attention_mask is not None:
                    attention_mask = inputs.attention_mask.to(DEVICE)
                with torch.no_grad():
                    generated_ids = asr_model.generate(
                        input_features=input_features,
                        attention_mask=attention_mask,
                        use_cache=False,
                    )
                return asr_processor.batch_decode(generated_ids, skip_special_tokens=True)

            result = await asyncio.to_thread(run_inference)
            text = (result[0] if result else "").strip()
            await self.write_event(Transcript(text=text).event())

        except Exception as e:
            _LOGGER.error("Transcription failed: %s", e, exc_info=True)


def set_last_requested_language(client_key: str, language: str) -> None:
    """Store the most recent language requested by an STT client."""
    store_dir = _get_language_store_dir()
    os.makedirs(store_dir, exist_ok=True)
    payload = {"language": language, "ts": time.time()}
    with open(_language_file_path(store_dir, client_key), "w", encoding="utf-8") as handle:
        json.dump(payload, handle)
    _prune_stale_languages(store_dir, payload["ts"])


def get_last_requested_language_for_client(client_key: str) -> str | None:
    """Return the most recent language requested by an STT client."""
    store_dir = _get_language_store_dir()
    data = _read_language_payload(store_dir, client_key)
    if not data:
        return None
    if _is_stale(data.get("ts")):
        _remove_language_file(store_dir, client_key)
        return None
    return data.get("language")


def _read_language_payload(store_dir: str, client_key: str) -> dict | None:
    path = _language_file_path(store_dir, client_key)
    if not os.path.isfile(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if isinstance(payload, dict):
            return payload
    except Exception:
        return None
    return None


def _remove_language_file(store_dir: str, client_key: str) -> None:
    path = _language_file_path(store_dir, client_key)
    try:
        if os.path.isfile(path):
            os.remove(path)
    except Exception:
        pass


def _prune_stale_languages(store_dir: str, now: float) -> None:
    try:
        for filename in os.listdir(store_dir):
            if not filename.startswith("wyoming_lang_") or not filename.endswith(".json"):
                continue
            path = os.path.join(store_dir, filename)
            try:
                with open(path, "r", encoding="utf-8") as handle:
                    payload = json.load(handle)
                ts = payload.get("ts")
            except Exception:
                ts = None
            if _is_stale(ts, now):
                try:
                    os.remove(path)
                except Exception:
                    pass
    except FileNotFoundError:
        return


def _is_stale(ts: float | None, now: float | None = None) -> bool:
    if not isinstance(ts, (int, float)):
        return True
    current = now if now is not None else time.time()
    return current - ts > _LANGUAGE_TTL_SEC


def _get_language_store_dir() -> str:
    return str(get_log_file_path().parent)


def _language_file_path(store_dir: str, client_key: str) -> str:
    safe_key = "".join(c if c.isalnum() or c in (".", "-", "_") else "_" for c in client_key)
    return os.path.join(store_dir, f"wyoming_lang_{safe_key}.json")


async def main():
    load_model()
    host = get_wyoming_stt_host()
    port = get_wyoming_stt_port()
    server = AsyncServer.from_uri(f"tcp://{host}:{port}")
    _LOGGER.info("Wyoming STT Server running on %s:%s", host, port)
    try:
        await server.run(WyomingSTTHandler)
    except KeyboardInterrupt:
        pass


def start_wyoming_stt_server_in_background() -> threading.Thread:
    """Start the Wyoming STT server in a background thread."""
    def runner():
        try:
            asyncio.run(main())
        except Exception as exc:
            _LOGGER.error("Wyoming STT server terminated unexpectedly: %s", exc, exc_info=True)

    thread = threading.Thread(target=runner, name="wyoming-stt-server", daemon=True)
    thread.start()
    return thread


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass

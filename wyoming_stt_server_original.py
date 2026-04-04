import asyncio
import io
import logging
import threading

import numpy as np
import torch
import whisper

from wyoming.info import Describe, Info, AsrProgram, AsrModel, Attribution
from wyoming.server import AsyncEventHandler, AsyncServer
from wyoming.audio import AudioChunk, AudioStart, AudioStop
from wyoming.asr import Transcribe, Transcript

# CONFIGURATION
# ---------------------
PORT = 10300  # Default Wyoming STT port
MODEL_SIZE = "small"  # Options: tiny, base, small, medium, large-v3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LANGUAGE = "tr"  # Force Turkish
# ---------------------

logging.basicConfig(level=logging.INFO)
_LOGGER = logging.getLogger(__name__)

# Global model variable
whisper_model = None

def load_model():
    """Load the Whisper model once at startup."""
    global whisper_model
    _LOGGER.info(f"Loading Whisper model '{MODEL_SIZE}' on {DEVICE}...")
    whisper_model = whisper.load_model(MODEL_SIZE, device=DEVICE)
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
            _LOGGER.info("Received AudioStart from client")

        elif AudioChunk.is_type(event.type):
            if self.is_receiving:
                chunk = AudioChunk.from_event(event)
                self.audio_buffer.write(chunk.audio)

        elif AudioStop.is_type(event.type):
            self.is_receiving = False
            _LOGGER.info("Received AudioStop from client. Transcribing...")
            await self._transcribe()

        elif Transcribe.is_type(event.type):
            _LOGGER.info("Received Transcribe request from client: %s", self._event_summary(event))

        elif Describe.is_type(event.type):
            _LOGGER.info("Received Describe request from client: %s", self._event_summary(event))
            await self._handle_describe()

        return True

    @staticmethod
    def _event_summary(event) -> str:
        """Best-effort summary of a Wyoming event for logging."""
        event_data = getattr(event, "data", None)
        if event_data is None:
            return f"type={getattr(event, 'type', 'unknown')}"
        return f"type={getattr(event, 'type', 'unknown')}, data={event_data}"

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
                            name=MODEL_SIZE,
                            description=f"Whisper {MODEL_SIZE} (Turkish)",
                            attribution=Attribution(name="OpenAI", url="https://github.com/openai/whisper"),
                            installed=True,
                            # --- CHANGE THIS LINE ---
                            languages=["tr-TR","en-GB"],  
                            # ------------------------
                            version="1.0.0"
                        )
                    ],
                )
            ]
        )
        await self.write_event(info.event())
        _LOGGER.info("Sent Info/Describe packet to Home Assistant")
                
    async def _transcribe(self):
        """Process the buffered audio and send text back."""
        global whisper_model

        # 1. Get bytes
        data = self.audio_buffer.getvalue()
        if not data:
            _LOGGER.warning("No audio data received.")
            return

        # 2. Convert Raw PCM (16-bit, 16kHz) to Float32 (-1.0 to 1.0)
        # Home Assistant sends 16kHz, 16-bit mono by default
        audio_np = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0

        # 3. Transcribe (Run in thread to not block the server loop)
        try:
            def run_inference():
                return whisper_model.transcribe(
                    audio_np, 
                    language=LANGUAGE,
                    fp16=(DEVICE == "cuda")
                )

            # Run in thread
            result = await asyncio.to_thread(run_inference)
            text = result["text"].strip()
            
            _LOGGER.info(f"Transcription: '{text}'")

            # 4. Send Transcript back to HA
            await self.write_event(Transcript(text=text).event())

        except Exception as e:
            _LOGGER.error(f"Transcription failed: {e}", exc_info=True)


async def main():
    load_model()
    server = AsyncServer.from_uri(f"tcp://0.0.0.0:{PORT}")
    _LOGGER.info(f"Wyoming STT Server running on 0.0.0.0:{PORT}")
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
            _LOGGER.error(f"Wyoming STT server terminated unexpectedly: {exc}", exc_info=True)

    thread = threading.Thread(target=runner, name="wyoming-stt-server", daemon=True)
    thread.start()
    return thread

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass

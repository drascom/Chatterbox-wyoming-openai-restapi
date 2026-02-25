import asyncio
import logging
import warnings
from contextlib import asynccontextmanager
from typing import Optional

import numpy as np
import torch
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import JSONResponse
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
import uvicorn


MODEL_ID = "selimc/whisper-large-v3-turbo-turkish"
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
TORCH_DTYPE = torch.float16 if torch.cuda.is_available() else torch.float32
DEFAULT_LANGUAGE = "tr"
SERVER_HOST = "0.0.0.0"
SERVER_PORT = 10400

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)
warnings.filterwarnings(
    "ignore",
    message=".*LoRACompatibleLinear.*",
    category=FutureWarning,
)

asr_model = None
asr_processor = None


def load_model() -> None:
    global asr_model, asr_processor
    logger.info("Loading Whisper model '%s' on %s...", MODEL_ID, DEVICE)
    asr_model = AutoModelForSpeechSeq2Seq.from_pretrained(
        MODEL_ID,
        torch_dtype=TORCH_DTYPE,
        low_cpu_mem_usage=True,
        use_safetensors=True,
    )
    asr_model.to(DEVICE)
    asr_model.eval()
    asr_processor = AutoProcessor.from_pretrained(MODEL_ID)
    logger.info("Whisper model loaded.")


def _resample_if_needed(audio: np.ndarray, src_rate: int, target_rate: int) -> np.ndarray:
    if src_rate == target_rate:
        return audio
    try:
        import librosa

        return librosa.resample(y=audio, orig_sr=src_rate, target_sr=target_rate)
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported sample_rate={src_rate}. Resampling failed: {exc}",
        ) from exc


def _pcm_to_mono_float32(raw: bytes, width: int, channels: int) -> np.ndarray:
    if width == 2:
        audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    elif width == 1:
        audio = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    elif width == 4:
        audio = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0
    else:
        raise HTTPException(status_code=400, detail=f"Unsupported PCM width: {width}")

    if channels > 1:
        frame_count = len(audio) // channels
        if frame_count == 0:
            return np.array([], dtype=np.float32)
        audio = audio[: frame_count * channels].reshape(frame_count, channels).mean(axis=1)

    return audio


async def transcribe_audio(audio: np.ndarray, language: Optional[str]) -> str:
    if asr_model is None or asr_processor is None:
        raise HTTPException(status_code=503, detail="ASR model is not loaded")

    def run_inference() -> str:
        inputs = asr_processor(
            audio,
            sampling_rate=16000,
            return_tensors="pt",
            return_attention_mask=True,
        )
        input_features = inputs.input_features.to(DEVICE, dtype=asr_model.dtype)
        attention_mask = None
        if hasattr(inputs, "attention_mask") and inputs.attention_mask is not None:
            attention_mask = inputs.attention_mask.to(DEVICE)

        generate_kwargs = {
            "input_features": input_features,
            "attention_mask": attention_mask,
            "use_cache": False,
            "task": "transcribe",
        }
        if language:
            generate_kwargs["language"] = language.split("-")[0]

        with torch.no_grad():
            generated_ids = asr_model.generate(**generate_kwargs)

        decoded = asr_processor.batch_decode(generated_ids, skip_special_tokens=True)
        return (decoded[0] if decoded else "").strip()

    return await asyncio.to_thread(run_inference)


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_model()
    yield


app = FastAPI(
    title="Whisper STT Internal API",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health")
async def health() -> JSONResponse:
    ready = asr_model is not None and asr_processor is not None
    return JSONResponse(
        {
            "status": "ok" if ready else "loading",
            "model": MODEL_ID,
            "device": DEVICE,
        },
        status_code=200 if ready else 503,
    )


@app.post("/internal/stt/transcribe")
async def transcribe_endpoint(
    request: Request,
    sample_rate: int = Query(16000, ge=1),
    channels: int = Query(1, ge=1),
    language: Optional[str] = Query(None),
    pcm_width: int = Query(2, ge=1),
) -> JSONResponse:
    raw = await request.body()
    if not raw:
        raise HTTPException(status_code=400, detail="Empty audio body")

    audio_mono = _pcm_to_mono_float32(raw, pcm_width, channels)
    if audio_mono.size == 0:
        raise HTTPException(status_code=400, detail="No decodable audio frames")

    if sample_rate != 16000:
        audio_mono = _resample_if_needed(audio_mono, sample_rate, 16000)

    lang = language or DEFAULT_LANGUAGE
    text = await transcribe_audio(audio_mono, lang)

    return JSONResponse(
        {
            "text": text,
            "model": MODEL_ID,
            "language": lang,
        }
    )


if __name__ == "__main__":
    uvicorn.run(
        "stt_http_server:app",
        host=SERVER_HOST,
        port=SERVER_PORT,
        log_level="info",
        workers=1,
        reload=False,
    )

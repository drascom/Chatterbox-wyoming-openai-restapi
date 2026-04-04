import asyncio
import io
import logging
import subprocess
import time
import warnings
from contextlib import asynccontextmanager
from typing import Literal, Optional

import numpy as np
import torch
from fastapi import FastAPI, File, Form, HTTPException, Query, Request, Response, UploadFile
from fastapi.responses import JSONResponse, PlainTextResponse
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
import uvicorn


MODEL_ID = "selimc/whisper-large-v3-turbo-turkish"
MODEL_ALIASES = ("whisper-1",)
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
TORCH_DTYPE = torch.float16 if torch.cuda.is_available() else torch.float32
DEFAULT_LANGUAGE = "tr"
SERVER_HOST = "0.0.0.0"
SERVER_PORT = 10400
ResponseFormat = Literal["text", "json", "verbose_json", "srt", "vtt"]

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


def _decode_with_ffmpeg(raw: bytes, sample_rate: int = 16000) -> np.ndarray:
    """Decode arbitrary media bytes via ffmpeg to mono PCM16."""
    ffmpeg_cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        "pipe:0",
        "-f",
        "s16le",
        "-acodec",
        "pcm_s16le",
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        "pipe:1",
    ]
    try:
        proc = subprocess.run(
            ffmpeg_cmd,
            input=raw,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=500, detail="ffmpeg is not installed on server") from exc
    except subprocess.CalledProcessError as exc:
        err = exc.stderr.decode("utf-8", errors="ignore").strip() or "ffmpeg decode failed"
        raise HTTPException(status_code=400, detail=f"Failed to decode audio with ffmpeg: {err}") from exc

    pcm = np.frombuffer(proc.stdout, dtype=np.int16).astype(np.float32) / 32768.0
    if pcm.size == 0:
        raise HTTPException(status_code=400, detail="Decoded audio is empty")
    return pcm


def _decode_uploaded_audio(raw: bytes, sample_rate: int = 16000) -> np.ndarray:
    """Decode common formats; explicitly supports audio/webm (Opus) via ffmpeg fallback."""
    if not raw:
        raise HTTPException(status_code=400, detail="Empty audio file")
    try:
        import librosa

        audio, _ = librosa.load(io.BytesIO(raw), sr=sample_rate, mono=True)
        audio_np = np.asarray(audio, dtype=np.float32)
        if audio_np.size == 0:
            raise HTTPException(status_code=400, detail="Decoded audio is empty")
        return audio_np
    except HTTPException:
        raise
    except Exception:
        # Some containers/codecs (notably audio/webm + Opus) can fail in librosa.
        # ffmpeg handles these reliably when available in the runtime image.
        return _decode_with_ffmpeg(raw, sample_rate=sample_rate)


def _normalize_model_id(model: str) -> str:
    candidate = (model or "").strip()
    if not candidate or candidate == MODEL_ID or candidate in MODEL_ALIASES:
        return MODEL_ID
    raise HTTPException(status_code=404, detail=f"Model '{candidate}' not found")


def _model_payload(model_id: str) -> dict:
    return {
        "id": model_id,
        "object": "model",
        "created": int(time.time()),
        "owned_by": "openai",
        "root": MODEL_ID,
    }


def _format_timestamp(total_seconds: float, *, srt: bool) -> str:
    total_millis = max(0, int(round(total_seconds * 1000)))
    hours, remainder = divmod(total_millis, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    seconds, millis = divmod(remainder, 1000)
    if srt:
        return f"{hours:02}:{minutes:02}:{seconds:02},{millis:03}"
    return f"{hours:02}:{minutes:02}:{seconds:02}.{millis:03}"


def _format_transcript_response(
    *,
    text: str,
    model: str,
    language: Optional[str],
    task: Literal["transcribe", "translate"],
    response_format: ResponseFormat,
    duration_seconds: float,
) -> Response:
    if response_format == "text":
        return PlainTextResponse(text)
    if response_format == "verbose_json":
        return JSONResponse(
            {
                "task": task,
                "language": language or "",
                "duration": round(duration_seconds, 3),
                "text": text,
                "segments": [],
                "words": [],
            }
        )
    if response_format == "srt":
        srt = f"1\n{_format_timestamp(0, srt=True)} --> {_format_timestamp(duration_seconds, srt=True)}\n{text}\n"
        return Response(content=srt, media_type="text/plain")
    if response_format == "vtt":
        vtt = f"WEBVTT\n\n{_format_timestamp(0, srt=False)} --> {_format_timestamp(duration_seconds, srt=False)}\n{text}\n"
        return Response(content=vtt, media_type="text/vtt")
    return JSONResponse({"text": text, "model": model, "language": language})


async def transcribe_audio(audio: np.ndarray, language: Optional[str], task: Literal["transcribe", "translate"]) -> str:
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
            "task": task,
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
    text = await transcribe_audio(audio_mono, lang, "transcribe")

    return JSONResponse(
        {
            "text": text,
            "model": MODEL_ID,
            "language": lang,
        }
    )


@app.post("/stt")
async def public_stt_endpoint(
    file: UploadFile = File(...),
    language: Optional[str] = Form(None),
) -> JSONResponse:
    raw = await file.read()
    audio_mono = _decode_uploaded_audio(raw, sample_rate=16000)
    lang = language or DEFAULT_LANGUAGE
    text = await transcribe_audio(audio_mono, lang, "transcribe")
    return JSONResponse({"text": text, "model": MODEL_ID, "language": lang})


@app.post("/v1/audio/transcriptions", response_model=None)
async def openai_transcriptions_endpoint(
    file: UploadFile = File(...),
    model: str = Form(MODEL_ID),
    language: Optional[str] = Form(None),
    prompt: Optional[str] = Form(None),
    response_format: ResponseFormat = Form("json"),
    temperature: float = Form(0.0),
    timestamp_granularities: Optional[list[str]] = Form(None, alias="timestamp_granularities[]"),
    stream: bool = Form(False),
    hotwords: Optional[str] = Form(None),
    without_timestamps: bool = Form(True),
):
    del prompt, temperature, timestamp_granularities, hotwords, without_timestamps
    selected_model = _normalize_model_id(model)
    raw = await file.read()
    audio_mono = _decode_uploaded_audio(raw, sample_rate=16000)
    lang = language or DEFAULT_LANGUAGE
    if stream:
        logger.warning("`stream=true` requested on /v1/audio/transcriptions, but streaming is not implemented; returning a standard response")
    text = await transcribe_audio(audio_mono, lang, "transcribe")
    return _format_transcript_response(
        text=text,
        model=selected_model,
        language=lang,
        task="transcribe",
        response_format=response_format,
        duration_seconds=len(audio_mono) / 16000.0,
    )


@app.post("/v1/audio/translations", response_model=None)
async def openai_translations_endpoint(
    file: UploadFile = File(...),
    model: str = Form(MODEL_ID),
    prompt: Optional[str] = Form(None),
    response_format: ResponseFormat = Form("json"),
    temperature: float = Form(0.0),
):
    del prompt, temperature
    selected_model = _normalize_model_id(model)
    raw = await file.read()
    audio_mono = _decode_uploaded_audio(raw, sample_rate=16000)
    text = await transcribe_audio(audio_mono, None, "translate")
    return _format_transcript_response(
        text=text,
        model=selected_model,
        language="en",
        task="translate",
        response_format=response_format,
        duration_seconds=len(audio_mono) / 16000.0,
    )


@app.get("/v1/models")
async def list_models() -> JSONResponse:
    return JSONResponse({"object": "list", "data": [_model_payload(MODEL_ID), *[_model_payload(alias) for alias in MODEL_ALIASES]]})


@app.get("/v1/models/{model_id:path}")
async def get_model(model_id: str) -> JSONResponse:
    normalized_id = _normalize_model_id(model_id)
    if model_id in MODEL_ALIASES:
        return JSONResponse(_model_payload(model_id))
    return JSONResponse(_model_payload(normalized_id))


if __name__ == "__main__":
    uvicorn.run(
        "stt_http_server:app",
        host=SERVER_HOST,
        port=SERVER_PORT,
        log_level="info",
        workers=1,
        reload=False,
    )

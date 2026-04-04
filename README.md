# Chatterbox-TR-Api (fork of devnen/Chatterbox-TTS-Server)

FastAPI server for the Chatterbox TTS model. This fork keeps the upstream UX/features and adds:
- Default config tuned for Turkish as well as English (language selector is for model input; the UI itself remains English).
- OpenAI-compatible `/v1/audio/speech` endpoint.
- Split deployment with dedicated containers for TTS, STT, and optional Wyoming gateway.
- Tested with Python 3.11 (recommended to avoid dependency pinning issues, e.g., protobuf).

## Install (bare metal)
```
uv venv
source venv/bin/activate
uv pip install -r requirements.txt
uv run python server.py
```
Adjust `config.yaml` for paths, defaults, and audio settings; it auto-creates on first run.

## FastAPI / General Use
### Compose (recommended)
Build images first (required after code changes):
```
docker compose build
```

Run TTS + STT only:
```
docker compose up -d chatterbox-tts whisper-stt
```

Run TTS + STT + Wyoming gateway:
```
COMPOSE_PROFILES=wyoming docker compose up -d
```

You can also build and start in one step:
```
docker compose up -d --build
```

The compose file uses Docker internal DNS between services:
- `chatterbox-tts` serves FastAPI/UI on container port `8000` and host port `${PORT:-8001}`.
- `whisper-stt` serves STT HTTP API on container port `8001` and host port `${STT_API_PORT:-8000}`.
- `wyoming-gateway` publishes Wyoming TTS/STT on host `${WYOMING_PORT:-10200}` and `${WHISPER_PORT:-10300}`.

### Docker run (single role examples)
TTS API/UI container:
```
docker run --rm --gpus all \
  -p 8001:8000 \
  -v $PWD/config.yaml:/app/config.yaml \
  -v $PWD/outputs:/app/outputs \
  -v $PWD/voices:/app/voices \
  -v $PWD/reference_audio:/app/reference_audio \
  drascom07/chatterbox-wyoming-openai-restapi:latest \
  python3 server.py
```

### Development
When developing inside this repo, rebuild the image locally so you can test your changes:
```
docker build -t chatterbox-wyoming-openai-restapi .
docker run --rm --gpus all -p 8001:8000 \
  -v $PWD/config.yaml:/app/config.yaml \
  -v $PWD/outputs:/app/outputs \
  -v $PWD/voices:/app/voices \
  -v $PWD/reference_audio:/app/reference_audio \
  chatterbox-wyoming-openai-restapi \
  python3 server.py
```
You can also run the compose file with a rebuild instead of `pull`:
```
docker compose up -d --build
```
For a clean rebuild (removes images/volumes/orphans):
```
docker compose down --rmi all --volumes --remove-orphans
docker system prune -a --volumes -f
```

## Home Assistant (Wyoming)
- Start the optional `wyoming-gateway` service with profile `wyoming`.
- Publish ports when running in Docker (default host ports): TTS `10200`, STT `10300`.
- In Home Assistant, add the “Wyoming Protocol” integration pointing to the host/port for each service.

Wyoming TTS (Chatterbox):
- Default port: `10200` (`WYOMING_PORT` overrides host side).
- Provider name in HA: “Chatterbox TTS (Wyoming)”.
- Gateway calls TTS upstream via `TTS_UPSTREAM_URL` (default: `http://chatterbox-tts:8000`).

Wyoming STT (Whisper):
- Default port: `10300` (`WHISPER_PORT` overrides host side).
- Gateway calls STT upstream via `STT_UPSTREAM_URL` (default: `http://whisper-stt:10400`).
- Whisper model is loaded in `stt_http_server.py` and currently set to `selimc/whisper-large-v3-turbo-turkish`.

## Quick curl tests (deployed server)
Set your base URL (example uses your deployment):
```
BASE_URL="https://chatter.drascom.uk"
```

Basic checks:
```
curl -i "$BASE_URL/"
curl -i "$BASE_URL/health"
curl -sS "$BASE_URL/api/model-status"
curl -i "$BASE_URL/docs"
curl -i "$BASE_URL/openapi.json"
```

Simple TTS POST (`/tts`, JSON):
```
curl -sS -X POST "$BASE_URL/tts" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Merhaba dunya, bu bir testtir.",
    "voice_mode": "predefined",
    "predefined_voice_id": "Abigail.wav",
    "output_format": "wav",
    "language": "tr"
  }' \
  -o tts_output.wav
```

OpenAI-compatible TTS POST (`/v1/audio/speech`, JSON):
```
curl -sS -X POST "$BASE_URL/v1/audio/speech" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "chatterbox",
    "input": "Hello from Chatterbox API",
    "voice": "Abigail.wav",
    "response_format": "wav"
  }' \
  -o speech_output.wav
```

STT POST (`/v1/audio/transcriptions`, multipart, STT service URL):
```
STT_BASE_URL="http://localhost:8000"
curl -sS -X POST "$STT_BASE_URL/v1/audio/transcriptions" \
  -F "file=@sample.wav" \
  -F "model=whisper-1" \
  -F "language=tr" \
  -F "response_format=json"
```

## External Whisper STT API
Whisper STT container also exposes public HTTP endpoints for external callers:

- `POST /stt` (multipart form-data)
  - fields:
    - `file` (audio file: wav/mp3/ogg/flac/opus and audio/webm+opus supported)
    - `language` (optional, default `tr`)
  - response:
    - `{"text":"...","model":"selimc/whisper-large-v3-turbo-turkish","language":"tr"}`

- `POST /v1/audio/transcriptions` (OpenAI-style multipart form-data)
  - fields:
    - `file` (required)
    - `model` (required by OpenAI-style clients; accepts `whisper-1` or the server model id)
    - `language` (optional)
    - `prompt` (optional, accepted for compatibility)
    - `temperature` (optional, accepted for compatibility)
    - `response_format` (`json`, `text`, `verbose_json`, `srt`, or `vtt`)
    - `timestamp_granularities[]` (optional, accepted for compatibility)
    - `stream` (optional, accepted but currently returns a normal non-streaming response)
  - response:
    - `json`: `{"text":"...","model":"...","language":"..."}`
    - `text`: plain transcript text

- `POST /v1/audio/translations` (OpenAI-style multipart form-data)
  - fields:
    - `file` (required)
    - `model` (required by OpenAI-style clients; accepts `whisper-1` or the server model id)
    - `prompt` (optional, accepted for compatibility)
    - `temperature` (optional, accepted for compatibility)
    - `response_format` (`json`, `text`, `verbose_json`, `srt`, or `vtt`)
  - response:
    - `json`: `{"text":"...","model":"...","language":"en"}`
    - `text`: plain translated text

- `GET /v1/models`
  - response:
    - OpenAI-style list object with the server model and `whisper-1` alias

- `GET /v1/models/{model_id}`
  - accepts:
    - `whisper-1`
    - `selimc/whisper-large-v3-turbo-turkish`

Example:
```
curl -sS -X POST "http://localhost:8000/v1/audio/transcriptions" \
  -F "file=@sample.wav" \
  -F "model=whisper-1" \
  -F "language=tr" \
  -F "response_format=json"
```

Translation example:
```
curl -sS -X POST "http://localhost:8000/v1/audio/translations" \
  -F "file=@sample.wav" \
  -F "model=whisper-1" \
  -F "response_format=json"
```

Model list example:
```
curl -sS "http://localhost:8000/v1/models"
```

### Wyoming gateway environment variables
- `TTS_UPSTREAM_URL` (default `http://chatterbox-tts:8000`)
- `STT_UPSTREAM_URL` (default `http://whisper-stt:8001`)
- `WYOMING_TTS_HOST` (default `0.0.0.0`)
- `WYOMING_TTS_PORT` (default `10200`)
- `WYOMING_STT_HOST` (default `0.0.0.0`)
- `WYOMING_STT_PORT` (default `10300`)
- `WYOMING_ENABLE_TTS` (default `true`)
- `WYOMING_ENABLE_STT` (default `true`)
- `WYOMING_ADVERTISE_NAME` (default `Chatterbox TTS (Wyoming)`)


## Notes vs upstream
- Upstream: https://github.com/devnen/Chatterbox-TTS-Server
- This fork: Turkish defaults, OpenAI speech endpoint, split TTS/STT/Wyoming deployment, and minor config/UI tweaks.

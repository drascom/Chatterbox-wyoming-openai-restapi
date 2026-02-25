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
Run TTS + STT only:
```
docker compose up -d chatterbox-tts whisper-stt
```

Run TTS + STT + Wyoming gateway:
```
COMPOSE_PROFILES=wyoming docker compose up -d
```

The compose file uses Docker internal DNS between services:
- `chatterbox-tts` serves FastAPI/UI on container port `8000` and host port `${PORT:-8004}`.
- `whisper-stt` serves STT HTTP API on container port `10400` and host port `${STT_API_PORT:-10400}`.
- `wyoming-gateway` publishes Wyoming TTS/STT on host `${WYOMING_PORT:-10200}` and `${WHISPER_PORT:-10300}`.

### Docker run (single role examples)
TTS API/UI container:
```
docker run --rm --gpus all \
  -p 8004:8000 \
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
docker run --rm --gpus all -p 8004:8000 \
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

## External Whisper STT API
Whisper STT container also exposes public HTTP endpoints for external callers:

- `POST /stt` (multipart form-data)
  - fields:
    - `file` (audio file: wav/mp3/ogg/flac/opus supported through decoder stack)
    - `language` (optional, default `tr`)
  - response:
    - `{"text":"...","model":"selimc/whisper-large-v3-turbo-turkish","language":"tr"}`

- `POST /v1/audio/transcriptions` (OpenAI-style multipart form-data)
  - fields:
    - `file` (required)
    - `model` (optional, defaults to server model id)
    - `language` (optional)
    - `response_format` (`json` or `text`)
  - response:
    - `json`: `{"text":"...","model":"...","language":"..."}`
    - `text`: plain transcript text

Example:
```
curl -sS -X POST "http://localhost:10400/v1/audio/transcriptions" \
  -F "file=@sample.wav" \
  -F "model=whisper-1" \
  -F "language=tr" \
  -F "response_format=json"
```

### Wyoming gateway environment variables
- `TTS_UPSTREAM_URL` (default `http://chatterbox-tts:8000`)
- `STT_UPSTREAM_URL` (default `http://whisper-stt:10400`)
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

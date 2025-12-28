# Chatterbox-TR-Api (fork of devnen/Chatterbox-TTS-Server)

FastAPI server for the Chatterbox TTS model. This fork keeps the upstream UX/features and adds:
- Default config tuned for Turkish as well as English (language selector is for model input; the UI itself remains English).
- OpenAI-compatible `/v1/audio/speech` endpoint.
- Optional Wyoming protocol server for Home Assistant voice pipelines.
- Tested with Python 3.11 (recommended to avoid dependency pinning issues, e.g., protobuf).

## Install (bare metal)
```
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python server.py
```
Adjust `config.yaml` for paths, defaults, and audio settings; it auto-creates on first run.

## Docker
Use the published image for day-to-day usage:
```
docker run --rm --gpus all -p 8004:8004 -p 10200:10200 \
  -v $PWD/outputs:/app/outputs \
  -v $PWD/voices:/app/voices \
  -v $PWD/reference_audio:/app/reference_audio \
  drascom07/chatterbox-wyoming-openai-restapi:latest
```
When running via `docker run`, include `--gpus all` (and the same volumes/ports) so the container can see your NVIDIA driver; `docker compose up -d` already works because it defaults to the configured NVIDIA runtime even without that flag.
Alternatively the bundled `docker-compose.yml` is tuned for the published image: run `docker compose pull` followed by `docker compose up -d`. The compose file now maps both the API port and, by default, `10200` for Wyoming (`WYOMING_PORT` lets you override the host side); only start listening on that port if `wyoming.enabled: true` in the config.

When developing inside this repo, rebuild the image locally so you can test your changes:
```
docker build -t chatterbox-wyoming-openai-restapi .
docker run --rm --gpus all -p 8004:8004 -p 10200:10200 \
  -v $PWD/outputs:/app/outputs \
  -v $PWD/voices:/app/voices \
  -v $PWD/reference_audio:/app/reference_audio \
  chatterbox-wyoming-openai-restapi
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
- In `config.yaml`, set `wyoming.enabled: true`; adjust `wyoming.host/port/sample_rate` if needed (defaults: `0.0.0.0:10200`, 24 kHz PCM16 mono).
- Install dependencies and start the server (`python server.py`) or run the container with port `10200` published.
- In Home Assistant, add the “Wyoming Protocol” integration pointing to that host/port, then pick the “Chatterbox TTS (Wyoming)” provider in your voice pipeline. Voices are exposed by filename from `voices/` and `reference_audio/`.
- Optional: add `tts_engine.voice_language_map` entries (e.g., `"Abigail.wav": "en"`) so HA can display voices like `Abigail (EN)` and the server can infer language when HA omits it.

## Standalone Wyoming test
If you only need to test the Wyoming protocol from Home Assistant without bringing up the FastAPI UI, run the new `wyoming_standalone.py` script:
```
python wyoming_standalone.py
```
The script reads the same `config.yaml`, starts the Wyoming `AsyncServer`, and logs readiness; you can override the bind address/port with `--host`/`--port` or force it even when `wyoming.enabled` is false using `--force`. Point HA at the advertised host/port (e.g., `localhost:10200`), exercise `/status` or `/speak`, and stop the script when done with Ctrl+C.

## Notes vs upstream
- Upstream: https://github.com/devnen/Chatterbox-TTS-Server
- This fork: Turkish defaults, OpenAI speech endpoint, Wyoming protocol integration, minor config/UI tweaks. All other behavior matches upstream unless noted above.

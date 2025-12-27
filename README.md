# Chatterbox-TR-Api (fork of devnen/Chatterbox-TTS-Server)

FastAPI server for the Chatterbox TTS model. This fork keeps the upstream UX/features and adds:
- Default config tuned for Turkish as well as English (language selector is for model input; the UI itself remains English).
- OpenAI-compatible `/v1/audio/speech` endpoint.
- Optional Wyoming protocol server for Home Assistant voice pipelines.

## Install (bare metal)
```
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python server.py
```
Adjust `config.yaml` for paths, defaults, and audio settings; it auto-creates on first run.

## Docker
```
docker build -t chatterbox-tr-api .
docker run --gpus all -p 8004:8004 -p 10200:10200 -v $PWD/outputs:/app/outputs -v $PWD/voices:/app/voices -v $PWD/reference_audio:/app/reference_audio chatterbox-tr-api
```
Or use `docker compose up -d  --build` with the included files. Expose `10200` only if you enable Wyoming.
If you need a full clean before rebuilding (removes images/volumes/orphans):
```
docker compose down --rmi all --volumes --remove-orphans
docker system prune -a --volumes -f
```

## Home Assistant (Wyoming)
- In `config.yaml`, set `wyoming.enabled: true`; adjust `wyoming.host/port/sample_rate` if needed (defaults: `0.0.0.0:10200`, 24 kHz PCM16 mono).
- Install dependencies and start the server (`python server.py`) or run the container with port `10200` published.
- In Home Assistant, add the “Wyoming Protocol” integration pointing to that host/port, then pick the “Chatterbox TTS (Wyoming)” provider in your voice pipeline. Voices are exposed by filename from `voices/` and `reference_audio/`.

## Notes vs upstream
- Upstream: https://github.com/devnen/Chatterbox-TTS-Server
- This fork: Turkish defaults, OpenAI speech endpoint, Wyoming protocol integration, minor config/UI tweaks. All other behavior matches upstream unless noted above.

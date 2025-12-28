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
Use the published image for day-to-day usage:
```
docker run --rm --gpus all -p 8004:8004 -p 10200:10200 \
  -v $PWD/outputs:/app/outputs \
  -v $PWD/voices:/app/voices \
  -v $PWD/reference_audio:/app/reference_audio \
  drascom07/chatterbox-wyoming-openai-restapi:latest
```
Alternatively the bundled `docker-compose.yml` is tuned for the published image: run `docker compose pull` followed by `docker compose up -d`, and only publish `10200` when Wyoming is enabled. (The compose file embeds `PORT`/volume mounts and GPU hints.)

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

## Notes vs upstream
- Upstream: https://github.com/devnen/Chatterbox-TTS-Server
- This fork: Turkish defaults, OpenAI speech endpoint, Wyoming protocol integration, minor config/UI tweaks. All other behavior matches upstream unless noted above.

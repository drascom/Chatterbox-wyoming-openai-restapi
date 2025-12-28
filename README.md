# Chatterbox-TR-Api (fork of devnen/Chatterbox-TTS-Server)

FastAPI server for the Chatterbox TTS model. This fork keeps the upstream UX/features and adds:
- Default config tuned for Turkish as well as English (language selector is for model input; the UI itself remains English).
- OpenAI-compatible `/v1/audio/speech` endpoint.
- Optional Wyoming protocol server for Home Assistant voice pipelines.
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
### Daily Use
Use the published image for day-to-day usage:
```
docker run --rm --gpus all \
  -p 8004:8004 \
  -p 10200:10200 \
  -p 10300:10300 \
  -v $PWD/outputs:/app/outputs \
  -v $PWD/voices:/app/voices \
  -v $PWD/reference_audio:/app/reference_audio \
  drascom07/chatterbox-wyoming-openai-restapi:latest
```
When running via `docker run`, include `--gpus all` (and the same volumes/ports) so the container can see your NVIDIA driver; `docker compose up -d` already works because it defaults to the configured NVIDIA runtime even without that flag.
Alternatively the bundled `docker-compose.yml` is tuned for the published image: run `docker compose pull` followed by `docker compose up -d`. The compose file maps the API port by default and you can override host-side ports with `PORT`, `WYOMING_PORT`, and `WHISPER_PORT`.

### Development
When developing inside this repo, rebuild the image locally so you can test your changes:
```
docker build -t chatterbox-wyoming-openai-restapi .
docker run --rm --gpus all -p 8004:8004 \
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
- Enable Wyoming in `config.yaml` (`wyoming.enabled: true`). This turns on both services below.
- Publish ports when running in Docker (default host ports): TTS `10200`, STT `10300`.
- In Home Assistant, add the “Wyoming Protocol” integration pointing to the host/port for each service.

Wyoming TTS (Chatterbox):
- Default port: `10200` (`WYOMING_PORT` overrides host side).
- Provider name in HA: “Chatterbox TTS (Wyoming)”.
- Voices are exposed by filename from `voices/` and `reference_audio/`.
- Optional: add `tts_engine.voice_language_map` entries (e.g., `"Abigail.wav": "en"`) so HA can display voices like `Abigail (EN)` and the server can infer language when HA omits it.

Wyoming STT (Whisper):
- Default port: `10300` (`WHISPER_PORT` overrides host side).
- Whisper model is loaded in `wyoming_stt_server.py` and currently set to Turkish (`LANGUAGE="tr"`). Adjust there if you need a different language.


## Notes vs upstream
- Upstream: https://github.com/devnen/Chatterbox-TTS-Server
- This fork: Turkish defaults, OpenAI speech endpoint, Wyoming protocol integration, minor config/UI tweaks. All other behavior matches upstream unless noted above.

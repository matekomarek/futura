# futura
# Futura AI

A personal AI assistant for Arch Linux + Hyprland, powered by Groq.
My first assistant project, so please be kind! 😄 (More details are in the comments at the top of the code.)

## What it can do
- Open apps and type / press keys in them
- Search the web, open pages and click/type in a browser (via Playwright)
- Look at your screen and answer questions about it
- Control volume and media (`volume 30`, `next song`, `pause`, `workspace 3` are instant)
- Remember things between sessions
- Run bash commands and read/write files in its own folder

For desktop tasks it shows you a plan and waits for your approval (y/n).
Bash commands and typing into terminals always ask separately.

## Requirements
- Arch Linux with Hyprland 0.55 or newer
- Python 3
- `sudo pacman -S wtype grim playerctl`
- A free Groq API key: https://console.groq.com/

## Installation
```bash
python -m venv .venv
.venv/bin/pip install groq
# optional, for the browser features:
.venv/bin/pip install playwright
```

## API key
Pick one:
- `export GROQ_API_KEY='your_key'`
- or create `~/.config/futura.env` containing `GROQ_API_KEY=your_key` (`chmod 600` it)
- or paste it on line 108 (press Ctrl+F and search `API_KEY_HERE`). Never upload your key to GitHub!

## Run
```bash
.venv/bin/python futura_assist_v3.py
```
Or give it a single request: `.venv/bin/python futura_assist_v3.py "open firefox"`.
List the available models with `--models`.

## Roadmap
- [ ] Speech recognition

## Feedback
I tried to optimize it as well as I could. If you find anything that could be improved, please open an issue or leave a comment. Thanks!

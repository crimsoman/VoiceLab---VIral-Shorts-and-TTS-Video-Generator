# Troubleshooting

Find the message closest to what you're seeing.

---

### "Python was not found on this PC"

Python isn't installed, or it's installed but not on your PATH.

1. Go to https://www.python.org/downloads/
2. Download **Python 3.11** (not 3.13 — not fully supported by some AI
   packages yet)
3. Run the installer. On the **first screen**, tick the box that says
   **"Add python.exe to PATH"** — this is the #1 cause of this error.
4. Restart `setup.bat`.

If you already have Python but still see this: open Command Prompt and
type `python --version`. If that also fails, PATH isn't set — reinstall
Python and make sure to tick the PATH box this time.

---

### "Found Python, but it's an unsupported version"

You have Python installed, but it's older than 3.10 or newer than 3.12
(e.g. 3.13, or 3.9). Install Python 3.11 alongside it from
python.org — you don't need to remove the old one. `setup.bat` will find
the right one automatically next time.

---

### venv creation fails

Usually one of these:
- **Antivirus/Windows Defender blocked it.** Temporarily add an exception
  for the VoiceLab folder, or check Defender's "Protection history" for a
  blocked action.
- **Path has unusual characters** (e.g. the folder is inside a path with
  `#`, `%`, or non-English characters). Move the whole VoiceLab folder
  somewhere simple, like `C:\VoiceLab`, and rerun setup.
- **Not enough permissions.** Right-click `setup.bat` → "Run as
  administrator", then try again.

---

### pip install fails / packages fail to download

- **Check your internet connection** — setup already tests this, but a
  connection can drop mid-install on slower networks. Just rerun
  `setup.bat`; already-downloaded packages won't re-download.
- **Corporate/school network or VPN blocking PyPI.** Try a different
  network (e.g. mobile hotspot) if you're on a restricted network.
- **Antivirus scanning every file during install** can make it very slow
  or time out. Temporarily exclude the VoiceLab folder in your antivirus
  settings during setup.

---

### "Port 8080 is already in use"

Something else on your PC is using port 8080.
- If it's an earlier VoiceLab window that didn't close properly, find and
  close it (check the taskbar, or press `Ctrl+Shift+Esc` → find `python.exe`
  → End Task).
- If it's another app, close that app, or ask on the GitHub Issues page
  for a way to change VoiceLab's port.

---

### Ollama / AI script writing doesn't work

- Ollama is a separate free program VoiceLab talks to for AI writing
  features — everything else in the app works without it.
- Download it from https://ollama.com/download/windows, install it, then
  just relaunch VoiceLab (`start.bat`) — no need to rerun setup.
- If Ollama is installed but the app still can't reach it, make sure the
  Ollama app/icon is actually running in your system tray.
- No AI model downloaded yet? Open VoiceLab → Settings tab → it lists
  models that fit your PC's RAM/GPU and lets you download one with a
  click.

---

### Thinking mode reasons but never gives a final answer

This is a real limitation of reasoning models (Qwen3, DeepSeek-R1, Gemma
with thinking, etc.) under a small context window - they can spend their
entire token budget "thinking" and never reach the actual answer. Two
places control this, and **both may need raising**:

1. **VoiceLab's own setting**: Settings → Thinking Mode Context Size
   (default 16k) - raise this slider if you have RAM to spare.
2. **Ollama's own app setting**: open the Ollama app itself → Settings →
   Context length. This can act as an independent ceiling on top of
   VoiceLab's setting - confirmed by testing that raising it to 32k
   (Ollama's own slider) fixed responses that were failing even with
   VoiceLab's setting already raised.

If you don't have RAM to spare for a larger context, turn Thinking mode
off in the chat's + menu instead - it isn't required for the model to
answer, just for it to show its reasoning first.

### GPU not detected, but I have an NVIDIA GPU

- Setup detects your GPU using the `nvidia-smi` command, which comes with
  NVIDIA's official drivers. Update your GPU drivers from
  https://www.nvidia.com/Download/index.aspx, restart your PC, then rerun
  `setup.bat`.
- AMD and Intel GPUs aren't used for AI acceleration in this version —
  the app will run those features on CPU instead, which still works, just
  slower.

---

### Browser opens but shows a blank page or an error like "Unexpected token"

The server is still starting up (loading AI models can take 30–90 seconds
the very first time you ever run it, less after that). Wait a minute and
refresh the page. If it's still blank after 2 minutes, close the black
window and run `start.bat` again — check the black window for any red
error text and search that exact text in this file or the GitHub Issues
page.

---

### First-time AI feature use is very slow / seems frozen

The **first** time you use AI Thumbnail, AI Music, or Voice Clone, the app
downloads that specific AI model (up to a few GB) — this only happens
once per feature. You'll see download progress in the app. After that
first download, it's cached locally and starts instantly.

---

### Setup finished, but a specific tab in the app says "not available"

That's expected for optional heavy AI features (AI Thumbnail, AI Music,
Parler voice, Voice Clone) — they aren't installed by `setup.bat` on
purpose, to keep the initial install small and fast. Open VoiceLab →
Settings tab → find the feature → click the one-click install button
shown there.

---

### Still stuck?

Open an Issue on the GitHub repo with:
- What step failed (setup.bat step number, or what happened in start.bat)
- The exact red error text you saw
- Your Windows version and whether you have an NVIDIA GPU

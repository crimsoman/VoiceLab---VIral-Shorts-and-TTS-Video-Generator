# VoiceLab

**An all-in-one, 100% local AI video/voiceover studio.**
No cloud, no subscriptions, no API keys. Everything runs on your own PC.

VoiceLab helps you turn raw footage into finished short-form video clips —
AI voiceover scripts, text-to-speech, voice cloning, auto-captions with
emoji, background music, and one-click vertical export — all from a single
web page running on your own computer.

> Built by **Ashmit**. Code implementation with Claude (Anthropic).
> If you build on this, a credit/link back is appreciated but not required —
> see [LICENSE](./LICENSE).

---

## Is this for me?

- ✅ You're on **Windows 10 or 11**
- ✅ You want everything to run **locally and privately** — nothing leaves your PC
- ✅ You don't need a powerful GPU — it works on CPU too (just slower for
  some AI features)

## What you get, and what it needs

| Feature | Needs | Works without it? |
|---|---|---|
| Voiceover (Edge TTS / Kokoro) | Nothing extra | Always works |
| Captions + emoji, video export | Nothing extra (ffmpeg is auto-installed) | Always works |
| AI script/caption writing | [Ollama](https://ollama.com) + a local model | App still runs, just skips AI writing |
| AI thumbnail / AI music / voice clone | A one-click install from inside the app | App still runs, just skips those tabs |

The app **scans your PC's RAM/GPU automatically** and recommends AI models
that will actually run well on your hardware — you're never forced into one
specific model.

---

## Install (one time only)

1. **Download this repository.**
   Click the green **Code** button on GitHub → **Download ZIP** → extract it
   anywhere (e.g. your Desktop).

2. **Double-click `setup.bat`.**
   This is the only step that installs anything. It will:
   - Check you have a compatible Python version (and tell you exactly what
     to do if you don't)
   - Check free disk space and internet connection
   - Set up a private Python environment *inside this folder only* — it
     never touches anything else on your PC
   - Scan for an NVIDIA GPU and install the right version of the AI engine
     for your hardware automatically
   - Check for Ollama (used for AI script writing) and guide you if it's
     missing
   - Offer to add a Desktop / Start Menu shortcut, or auto-start on login

   This step needs internet access and can take 5–15 minutes depending on
   your connection. You'll see clear progress messages the whole way.

3. **When it says "Setup complete!", you're done.**

## Run VoiceLab (every time after)

Double-click **`start.bat`** (or the shortcut you created). A browser
window opens automatically at `http://localhost:8080`. Close the black
window to stop the app.

## Uninstall

Double-click **`uninstall.bat`**. It removes the Python environment and any
shortcuts. It will ask separately before deleting your exported videos or
downloaded AI models — your work is never deleted silently.

---

## Something not working?

Check **[TROUBLESHOOTING.md](./TROUBLESHOOTING.md)** first — it covers the
most common setup issues in plain English with exact fixes.

---

## For developers

- Backend: `server.py` (FastAPI, single file)
- Frontend: `index.html` (single page, no build step)
- The app's own Settings tab has a live dependency checker + one-click
  installer for the heavier optional AI stack (`/dependency-status` and
  `/install-dependency/{key}` endpoints in `server.py`)
- `requirements.txt` is intentionally pinned to match the versions the
  in-app installer uses, so nothing conflicts later

Contributions/forks welcome under the MIT license. A mention of the
original author is appreciated if you build something on top of this.

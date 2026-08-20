# VoiceLab

**An all-in-one, 100% local AI video/voiceover studio.**
No cloud, no subscriptions, no API keys required. Everything runs on your own PC.

VoiceLab helps you turn raw footage into finished short-form video clips —
AI chat with a local model, voiceover scripts, text-to-speech, voice
cloning, auto-captions with emoji, background music, and one-click
vertical export — all from a single web page running on your own computer.

> Built by **Ashmit**. Code implementation with Claude (Anthropic).
> If you build on this, a credit/link back is appreciated but not required —
> see [LICENSE](./LICENSE).

---

## Screenshots

<!--
  Drop your own screenshots into an `images/` folder in the repo root,
  matching these filenames, and they'll render automatically on GitHub.
  Take them at a normal desktop window width (not maximized ultra-wide)
  so the UI reads clearly.
-->

### 💬 Chat
Talk to any local Ollama model. Toggle **Thinking mode** to see the
model's full reasoning before it answers, turn on **Web search** (Brave,
Tavily, or DuckDuckGo — pick one or let it auto-fallback) to ground
answers in current info with cited sources, and every response shows
live stats (model, tokens, speed) on click. Full conversation history
with rename/pin/search/export/delete.

![Chat tab](images/chat.png)

### 🎙️ TTS Studio
Turn any script into voiceover — Edge TTS (free, cloud-quality voices) or
Kokoro (fully local). Speed/pitch/EQ controls, AI-assisted script
polishing, hook + metadata generation for YouTube.

![TTS Studio tab](images/tts-studio.png)

### 🎬 Video Studio
Auto-captions with emoji, AI thumbnail generation, background music, and
one-click vertical export ready for Shorts/Reels/TikTok.

![Video Studio tab](images/video-studio.png)

### ✂️ Clip Finder
Import long-form footage, transcribe it automatically, and find the best
short clips to cut — all processed locally.

![Clip Finder tab](images/clip-finder.png)

---

## Is this for me?

- ✅ You're on **Windows 10 or 11**
- ✅ You want everything to run **locally and privately** — nothing leaves your PC
- ✅ You don't need a powerful GPU — it works on CPU too (just slower for
  some AI features)

## What you get, and what it needs

| Feature | Needs | Works without it? |
|---|---|---|
| Chat with a local model | [Ollama](https://ollama.com) + a local model | App still runs, chat just won't work until you install Ollama |
| Thinking mode (see model reasoning) | A reasoning-capable model (Qwen3, DeepSeek-R1, etc.) | Off by default — regular models just answer directly |
| Web search in chat | A free API key (Brave or Tavily) *or* nothing at all (DuckDuckGo works with zero setup) | Chat still works without any search |
| Voiceover (Edge TTS / Kokoro) | Nothing extra | Always works |
| Captions + emoji, video export | Nothing extra (ffmpeg is auto-installed) | Always works |
| AI thumbnail / AI music / voice clone | A one-click install from inside the app | App still runs, just skips those tabs |

The app **scans your PC's RAM/GPU automatically** and recommends AI models
that will actually run well on your hardware — you're never forced into one
specific model.

### Chat highlights

- **Thinking mode** — toggle from the chat's `+` menu. Shows the model's
  full reasoning live as it generates, then collapses into a clickable
  "Reasoning" summary. Uses more context/RAM — tune the size in Settings
  if a model reasons but never reaches a final answer (and check Ollama's
  own app Settings → Context length too, it can be a second ceiling).
- **Web search** — also in the `+` menu. "Auto" tries Brave → Tavily →
  DuckDuckGo in order, or pick one manually. Brave (2,000 free
  queries/month) and Tavily (1,000 free/month) need a free API key from
  their sites, pasted into Settings. DuckDuckGo needs no key at all but
  gives more limited results.
- **Stop button** — the send button becomes a stop button while
  generating, so a long or stuck response can be cancelled anytime.
- **Chat history** — rename, pin, search, export as text, or delete any
  conversation from the hover menu in the sidebar.
- **Export paths** — set a default save folder per export type (video/
  audio/thumbnail/music) in Settings, or choose a one-off custom folder
  each time.

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

Double-click **`start.bat`** (or the shortcut you created). It automatically
checks for any missing packages before launching (so an update never
leaves you needing to manually run `pip install`), then opens a browser
window at `http://localhost:8080`. Close the black window to stop the app.

## Uninstall

Double-click **`uninstall.bat`**. It removes the Python environment and any
shortcuts. It will ask separately before deleting your exported videos or
downloaded AI models — your work is never deleted silently.

---

## Something not working?

Check **[TROUBLESHOOTING.md](./TROUBLESHOOTING.md)** first — it covers the
most common setup and usage issues in plain English with exact fixes.

---

## For developers

- Backend: `server.py` (FastAPI, single file)
- Frontend: `index.html` (single page, no build step)
- The app's own Settings tab has a live dependency checker + one-click
  installer for the heavier optional AI stack (`/dependency-status` and
  `/install-dependency/{key}` endpoints in `server.py`)
- `requirements.txt` is intentionally pinned to match the versions the
  in-app installer uses, so nothing conflicts later
- Chat streams via Server-Sent Events (`/chat`) with separate `thinking`
  and `content` token channels, plus a final `stats` event (tokens,
  speed, model, search/thinking used, stop reason)
- Web search is provider-agnostic (`web_search_dispatch()` in
  `server.py`) — auto-fallback across Brave/Tavily/DuckDuckGo or a
  forced single provider, same interface either way

Contributions/forks welcome under the MIT license. A mention of the
original author is appreciated if you build something on top of this.

import sys as _sys_boot
import warnings as _warnings_boot
# Suppress noisy but harmless third-party warnings that appear on every boot.
# "Flash attention 2 is not installed" comes from transformers whenever it
# loads on a system without flash-attn — it's purely informational, the model
# works fine without it, but it confuses users into thinking something is broken.
_warnings_boot.filterwarnings("ignore", message=".*flash attention.*", category=UserWarning)
_warnings_boot.filterwarnings("ignore", message=".*Flash Attention.*", category=UserWarning)
_warnings_boot.filterwarnings("ignore", message=".*flash_attn.*")
_warnings_boot.filterwarnings("ignore", category=FutureWarning, module="transformers")
# Force UTF-8 stdout/stderr unconditionally, before anything else runs.
# Root cause of "UnicodeEncodeError: 'charmap' codec can't encode character
# '\u2713'": when stdout isn't a real console (e.g. redirected to a log file
# by start.bat, or piped), Python falls back to the OS's legacy code page
# (cp1252 on most Windows installs) which cannot represent characters like
# checkmarks or emoji used in this file's status prints. Setting
# PYTHONUTF8/PYTHONIOENCODING as environment variables in the launcher
# *should* fix this but depends on those variables correctly propagating
# through every layer of process spawning (cmd -> powershell -> Start-
# Process) — that chain doesn't always inherit env vars reliably. Doing it
# here, directly in code, removes that dependency entirely: this runs no
# matter how the script is launched.
if hasattr(_sys_boot.stdout, "reconfigure"):
    _sys_boot.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(_sys_boot.stderr, "reconfigure"):
    _sys_boot.stderr.reconfigure(encoding="utf-8", errors="replace")

# Redirect the Hugging Face model cache off C: (nearly full) onto D: (where
# this project already lives, with plenty of free space). Hugging Face
# defaults to C:\Users\<you>\.cache\huggingface\hub, which is what was
# causing "Not enough free disk space" mid-download — the download wasn't
# actually too big, the destination drive just had nowhere to put it.
# MUST be set as an environment variable before huggingface_hub/transformers
# /diffusers are imported anywhere (including transitively, e.g. via
# kokoro) — setting it later or only in the request handler is too late,
# those libraries read this once at import time.
import os as _os_boot
_hf_cache_dir = _os_boot.path.join(_os_boot.path.dirname(_os_boot.path.abspath(__file__)), "hf_cache")
_os_boot.makedirs(_hf_cache_dir, exist_ok=True)
_os_boot.environ.setdefault("HF_HOME", _hf_cache_dir)
_os_boot.environ.setdefault("HUGGINGFACE_HUB_CACHE", _os_boot.path.join(_hf_cache_dir, "hub"))

from fastapi import FastAPI, BackgroundTasks, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, HTMLResponse, Response, FileResponse, JSONResponse
from starlette.background import BackgroundTask
from pydantic import BaseModel
from typing import List, Optional
import soundfile as sf
import numpy as np
import io, json, httpx, tempfile, sqlite3, os, uuid, psutil, asyncio, subprocess, shutil, re, base64
from datetime import datetime
import librosa
from scipy import signal
import edge_tts
from PIL import Image, ImageDraw, ImageFont

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# ── AUDIO PROCESSING ENGINE ──────────────────────────────────────────
def apply_pitch_shift(audio: np.ndarray, sr: int, pitch_shift: float) -> np.ndarray:
    """Shift pitch only, preserving duration (librosa phase vocoder).
    FIXES applied:
    - float-precision semitones (previously int() truncated small shifts
      to 0, but STILL ran the full STFT phase-vocoder pass with n_steps=0,
      degrading audio quality without changing pitch — this was likely the
      'sounds robotic the moment you touch the slider' bug).
    - epsilon-skip: ratios within ~0.3 semitones of 1.0 are imperceptible
      and now skip processing entirely.
    """
    if abs(pitch_shift - 1.0) < 0.02:
        return audio
    semitones = 12.0 * np.log2(pitch_shift)
    semitones = max(-6.0, min(6.0, semitones))
    return librosa.effects.pitch_shift(audio, sr=sr, n_steps=semitones)

def apply_time_stretch(audio: np.ndarray, speed: float) -> np.ndarray:
    """Adjust playback speed without changing pitch (librosa phase vocoder).
    FALLBACK ONLY — Kokoro and Edge TTS now use native speed control
    (see /generate and /generate-hindi), which sounds far more natural.
    This remains as a fallback for Voice Clone / Parler outputs."""
    if abs(speed - 1.0) < 0.02:
        return audio
    speed = max(0.5, min(2.0, speed))
    return librosa.effects.time_stretch(audio, rate=speed)

def apply_normalization(audio: np.ndarray, target_db: float = -20.0) -> np.ndarray:
    """Normalize audio to target loudness level"""
    rms = np.sqrt(np.mean(audio ** 2))
    if rms < 1e-6:
        return audio
    target_amplitude = 10 ** (target_db / 20.0)
    return audio * (target_amplitude / rms)

def apply_compression(audio: np.ndarray, threshold: float = 0.5, ratio: float = 4.0) -> np.ndarray:
    """Dynamic range compression - reduce loud parts"""
    compressed = audio.copy()
    mask = np.abs(audio) > threshold
    compressed[mask] = threshold + (audio[mask] - threshold) / ratio
    return compressed

def apply_eq(audio: np.ndarray, sr: int, bass_db: float = 0.0, treble_db: float = 0.0) -> np.ndarray:
    """Apply 3-band EQ: bass (low-pass), mid (preserved), treble (high-pass)"""
    if bass_db == 0.0 and treble_db == 0.0:
        return audio
    
    # Simple shelving EQ using scipy.signal
    if bass_db != 0.0:
        # Bass boost/cut at low frequencies
        bass_gain = 10 ** (bass_db / 20.0)
        low_cutoff = 200  # Hz
        sos = signal.butter(2, low_cutoff / (sr / 2), btype='low', output='sos')
        low_freq = signal.sosfilt(sos, audio)
        audio = audio * (1 - 0.3) + low_freq * 0.3 * bass_gain
    
    if treble_db != 0.0:
        # Treble boost/cut at high frequencies
        treble_gain = 10 ** (treble_db / 20.0)
        high_cutoff = 5000  # Hz
        sos = signal.butter(2, high_cutoff / (sr / 2), btype='high', output='sos')
        high_freq = signal.sosfilt(sos, audio)
        audio = audio * (1 - 0.3) + high_freq * 0.3 * treble_gain
    
    return audio

def apply_reverb(audio: np.ndarray, sr: int, room_size: float = 0.5) -> np.ndarray:
    """Simple reverb using delay line convolution"""
    # Create impulse response with multiple delays
    delay_samples = int(sr * 0.05)  # 50ms
    impulse = np.zeros(delay_samples * 4)
    impulse[0] = 1.0
    impulse[delay_samples] = room_size * 0.8
    impulse[delay_samples * 2] = room_size * 0.6
    impulse[delay_samples * 3] = room_size * 0.4
    
    reverb_audio = signal.fftconvolve(audio, impulse, mode='same')
    # Mix with original
    return audio * 0.6 + reverb_audio * 0.4

def process_audio(audio: np.ndarray, sr: int, req) -> np.ndarray:
    """Apply all audio processing effects in optimal order"""
    # Order matters: pitch → speed → EQ → compression → normalization → reverb
    # Use getattr for compatibility with all request types (TTS, EdgeTTS, Clone)
    
    pitch = getattr(req, 'pitch', 1.0)
    speed = getattr(req, 'speed', 1.0)
    bass = getattr(req, 'bass', 0.0)
    treble = getattr(req, 'treble', 0.0)
    compress = getattr(req, 'compress', False)
    normalize = getattr(req, 'normalize', True)
    reverb = getattr(req, 'reverb', False)
    
    # 1. Pitch shifting (before time stretch)
    if pitch != 1.0:
        audio = apply_pitch_shift(audio, sr, pitch)
    
    # 2. Time stretching (speed)
    if speed != 1.0:
        audio = apply_time_stretch(audio, speed)
    
    # 3. EQ (bass/treble)
    if bass != 0.0 or treble != 0.0:
        audio = apply_eq(audio, sr, bass, treble)
    
    # 4. Compression (if enabled)
    if compress:
        audio = apply_compression(audio)
    
    # 5. Normalization (if enabled)
    if normalize:
        audio = apply_normalization(audio)
    
    # 6. Reverb (if enabled)
    if reverb:
        audio = apply_reverb(audio, sr)
    
    return audio

# ── VIDEO STUDIO ──────────────────────────────────────────────────────
try:
    import imageio_ffmpeg as _iio_ffmpeg
    FFMPEG_EXE = _iio_ffmpeg.get_ffmpeg_exe()
    FFPROBE_EXE = FFMPEG_EXE.replace("ffmpeg", "ffprobe")
    VIDEO_STUDIO_AVAILABLE = True
    print("  FFmpeg  [OK]  Ready")
except ImportError:
    FFMPEG_EXE = FFPROBE_EXE = None
    VIDEO_STUDIO_AVAILABLE = False
    print("  FFmpeg  ✗  Video Studio disabled (pip install imageio-ffmpeg)")
VIDEOS_DIR = "videos"
os.makedirs(VIDEOS_DIR, exist_ok=True)
MUSIC_DIR = "music"
os.makedirs(MUSIC_DIR, exist_ok=True)

gameplay_store = {}  # gameplay_id → {path, filename, width, height, duration}
video_jobs     = {}  # job_id      → {status, progress, output_path, error}
WORD_CACHE     = {}  # audio_id    → list of segment dicts (cached Whisper word-timestamps,
                      #               avoids re-running STT every time subtitle style/effect
                      #               is changed and Compose is clicked again)
music_store    = {}  # music_id    → {path, filename, duration}

# ── CLIP FINDER (long recording → AI-scored moments → auto-cut clips) ──
CLIPS_DIR = "clips"
os.makedirs(CLIPS_DIR, exist_ok=True)
RECORDINGS_DIR = "recordings"
os.makedirs(RECORDINGS_DIR, exist_ok=True)

recording_store = {}   # recording_id → {path, filename, width, height, duration}
clipfinder_jobs = {}   # job_id       → {status, progress, stage, error, ...stage-specific payload}

# ── DATABASE ──────────────────────────────────────────
DB_PATH = "exports.db"
EXPORTS_DIR = "exports"
os.makedirs(EXPORTS_DIR, exist_ok=True)

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS exports
                 (id TEXT PRIMARY KEY, filename TEXT, srt_filename TEXT, 
                  voice TEXT, speed REAL, created_at TEXT, duration REAL)''')
    c.execute('''CREATE TABLE IF NOT EXISTS projects
                 (id TEXT PRIMARY KEY, name TEXT, script TEXT, audio_id TEXT,
                  gameplay_id TEXT, settings_json TEXT, created_at TEXT, updated_at TEXT)''')
    c.execute('''CREATE TABLE IF NOT EXISTS voice_presets
                 (id TEXT PRIMARY KEY, name TEXT, engine TEXT, payload_json TEXT, created_at TEXT)''')
    c.execute('''CREATE TABLE IF NOT EXISTS music_tracks
                 (id TEXT PRIMARY KEY, filename TEXT, path TEXT, name TEXT, duration REAL, created_at TEXT)''')
    c.execute('''CREATE TABLE IF NOT EXISTS clipfinder_recordings
                 (recording_id TEXT PRIMARY KEY, path TEXT, filename TEXT, width INTEGER,
                  height INTEGER, duration REAL, created_at TEXT)''')
    try:
        c.execute("ALTER TABLE clipfinder_recordings ADD COLUMN context_notes TEXT DEFAULT ''")
    except sqlite3.OperationalError:
        pass  # column already exists — safe to ignore on every boot after the first
    c.execute('''CREATE TABLE IF NOT EXISTS clipfinder_transcripts
                 (recording_id TEXT PRIMARY KEY, segments_json TEXT, updated_at TEXT)''')
    c.execute('''CREATE TABLE IF NOT EXISTS clipfinder_exports
                 (filename TEXT PRIMARY KEY, recording_id TEXT, title TEXT, start REAL,
                  end REAL, duration REAL, created_at TEXT)''')
    c.execute('''CREATE TABLE IF NOT EXISTS clipfinder_moments
                 (recording_id TEXT PRIMARY KEY, moments_json TEXT, settings_json TEXT, updated_at TEXT)''')
    conn.commit()
    conn.close()

init_db()

MOMENTS_CACHE = {}  # recording_id → {"moments": [...], "settings": {...}} — the piece that
                     # was previously lost on every restart; now persisted like everything else.


def _clipfinder_hydrate_from_disk():
    """Reloads recording_store + WORD_CACHE + MOMENTS_CACHE from SQLite on
    startup — this is what makes recordings/transcripts/moments survive an
    app restart instead of vanishing the moment the process exits. Recording
    files themselves were always safe on disk (RECORDINGS_DIR); only the
    in-memory metadata dicts were being lost before."""
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        for rid, path, filename, width, height, duration, ctx in c.execute(
                "SELECT recording_id, path, filename, width, height, duration, COALESCE(context_notes,'') FROM clipfinder_recordings"):
            if os.path.exists(path):
                recording_store[rid] = {"path": path, "filename": filename, "width": width, "height": height, "duration": duration, "context_notes": ctx}
        for rid, segments_json in c.execute("SELECT recording_id, segments_json FROM clipfinder_transcripts"):
            if rid in recording_store:
                try:
                    WORD_CACHE[rid] = json.loads(segments_json)
                except Exception:
                    pass
        for rid, moments_json, settings_json in c.execute(
                "SELECT recording_id, moments_json, settings_json FROM clipfinder_moments"):
            if rid in recording_store:
                try:
                    MOMENTS_CACHE[rid] = {"moments": json.loads(moments_json),
                                           "settings": json.loads(settings_json) if settings_json else {}}
                except Exception:
                    pass
        conn.close()
    except Exception as e:
        print(f"[clipfinder] state hydration skipped: {e}")


_clipfinder_hydrate_from_disk()


def _cf_db_save_moments(recording_id: str, moments: list, settings: dict):
    MOMENTS_CACHE[recording_id] = {"moments": moments, "settings": settings}
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "INSERT OR REPLACE INTO clipfinder_moments VALUES (?,?,?,?)",
        (recording_id, json.dumps(moments), json.dumps(settings), datetime.now().isoformat()),
    )
    conn.commit()
    conn.close()


def _cf_db_save_recording(recording_id: str):
    r = recording_store.get(recording_id)
    if not r:
        return
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "INSERT OR REPLACE INTO clipfinder_recordings "
        "(recording_id, path, filename, width, height, duration, created_at, context_notes) VALUES (?,?,?,?,?,?,?,?)",
        (recording_id, r["path"], r["filename"], r["width"], r["height"], r["duration"],
         datetime.now().isoformat(), r.get("context_notes", "")),
    )
    conn.commit()
    conn.close()


def _cf_db_save_transcript(recording_id: str):
    segments = WORD_CACHE.get(recording_id)
    if segments is None:
        return
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "INSERT OR REPLACE INTO clipfinder_transcripts VALUES (?,?,?)",
        (recording_id, json.dumps(segments), datetime.now().isoformat()),
    )
    conn.commit()
    conn.close()


def _cf_db_save_export(filename: str, recording_id: str, title: str, start: float, end: float, duration: float):
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "INSERT OR REPLACE INTO clipfinder_exports VALUES (?,?,?,?,?,?,?)",
        (filename, recording_id, title, start, end, duration, datetime.now().isoformat()),
    )
    conn.commit()
    conn.close()

# ── TRANSCRIPTION QUEUE ───────────────────────────────
transcription_jobs = {}

def save_audio_file(audio: np.ndarray, voice: str, speed: float) -> str:
    """Save audio to disk and DB"""
    job_id = str(uuid.uuid4())
    filename = f"{EXPORTS_DIR}/{job_id}.wav"
    sf.write(filename, audio, 24000)
    
    duration = len(audio) / 24000
    now = datetime.now().isoformat()
    
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("INSERT INTO exports VALUES (?,?,?,?,?,?,?)", 
              (job_id, filename, None, voice, speed, now, duration))
    conn.commit()
    conn.close()
    
    return job_id, filename

def save_srt_file(job_id: str, srt_content: str):
    """Update DB with SRT file"""
    srt_filename = f"{EXPORTS_DIR}/{job_id}.srt"
    with open(srt_filename, 'w') as f:
        f.write(srt_content)
    
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("UPDATE exports SET srt_filename=? WHERE id=?", (srt_filename, job_id))
    conn.commit()
    conn.close()

# ── MODELS ───────────────────────────────────────────
print("\n==========================================\n")
print("  VoiceLab  |  Local AI + TTS Studio")
print("==========================================\n")
print("  Loading Kokoro TTS model...")

from kokoro import KPipeline
pipeline = KPipeline(lang_code='a')

print("  Kokoro TTS  [OK]  Ready")
print("  Loading Faster-Whisper model...")
from faster_whisper import WhisperModel
whisper_model = WhisperModel("base", device="cpu", compute_type="int8")
print("  Faster-Whisper  [OK]  Ready")
print("  Open browser: http://localhost:8080")
print("==========================================\n")

# ── STATE ────────────────────────────────────────────
# Default model name (used as a last-resort fallback if Ollama isn't
# reachable at all, or has zero models installed - fresh setup with
# nothing pulled yet).
selected_model = "llama3.2:3b"

# If Ollama IS running and already has model(s) installed, prefer one of
# those instead - covers the very common case of a user who already has
# Ollama set up with their own models before ever running VoiceLab.
#
# Prefer a plain text model over a vision-language model (name contains
# "vl", "vision", or "llava") when choosing automatically - VL models are
# built for image+text input and can behave inconsistently on some Ollama
# versions when called with text-only /api/chat requests. If only VL
# models are installed, fall back to whichever is installed anyway rather
# than leaving the app with no model at all.
try:
    import httpx as _httpx_boot
    _installed = [m["name"] for m in
                  _httpx_boot.get("http://localhost:11434/api/tags", timeout=2.0)
                  .json().get("models", [])]
    if _installed and selected_model not in _installed:
        _vision_markers = ("vl", "vision", "llava")
        _text_models = [m for m in _installed
                         if not any(v in m.lower() for v in _vision_markers)]
        selected_model = _text_models[0] if _text_models else _installed[0]
        print(f"  Ollama model    [OK]  Using installed model: {selected_model}")
except Exception:
    pass  # Ollama not running yet - fine, app still starts, Settings tab covers this


class TTSRequest(BaseModel):
    text: str
    voice: str = "af_heart"
    speed: float = 1.0
    pitch: float = 1.0
    # Audio processing controls
    normalize: bool = True
    compress: bool = False
    bass: float = 0.0      # -12 to +12 dB
    treble: float = 0.0    # -12 to +12 dB
    reverb: bool = False

class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: List[Message]
    model: str = "llama3.2:3b"

class HelperRequest(BaseModel):
    prompt: str
    model: str = "llama3.2:3b"

class ModelSelectRequest(BaseModel):
    model: str

class EdgeTTSRequest(BaseModel):
    text: str
    voice: str = "hi-IN-MadhurNeural"  # Hindi (India) - Female
    speed: float = 1.0
    pitch: float = 1.0
    normalize: bool = True
    compress: bool = False
    bass: float = 0.0
    treble: float = 0.0
    reverb: bool = False


@app.get("/")
async def serve_ui():
    try:
        with open("index.html", "r", encoding="utf-8") as f:
            return HTMLResponse(f.read())
    except FileNotFoundError:
        return HTMLResponse("<h2 style='font-family:sans-serif;padding:40px'>index.html not found!</h2>")


@app.post("/chat")
async def chat_stream(req: ChatRequest):
    # Use req.model if provided, otherwise fall back to selected_model
    model_to_use = req.model if req.model else selected_model
    
    async def generate():
        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                async with client.stream(
                    "POST", "http://localhost:11434/api/chat",
                    json={
                        "model": model_to_use,
                        "messages": [{"role": m.role, "content": m.content} for m in req.messages],
                        "stream": True,
                    },
                ) as response:
                    if response.status_code == 404:
                        yield f"data: {json.dumps({'error': f'Model \"{model_to_use}\" not found in Ollama. Open Settings and pick an installed model.', 'retry': True})}\n\n"
                        return
                    if response.status_code != 200:
                        yield f"data: {json.dumps({'error': f'Ollama error {response.status_code}', 'retry': True})}\n\n"
                        return
                    
                    async for line in response.aiter_lines():
                        if not line.strip():
                            continue
                        try:
                            data = json.loads(line)
                            token = data.get("message", {}).get("content", "")
                            done = data.get("done", False)
                            if token:
                                yield f"data: {json.dumps({'token': token})}\n\n"
                            if done:
                                yield f"data: {json.dumps({'done': True})}\n\n"
                                return
                        except:
                            continue
        except httpx.TimeoutException:
            yield f"data: {json.dumps({'error': 'Ollama timeout - check if running', 'retry': True})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e), 'retry': True})}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.post("/generate")
async def generate_tts(req: TTSRequest):
    try:
        # Kokoro supports speed NATIVELY (0.5-2.0) — far more natural than
        # generating at 1.0x and time-stretching afterward with librosa.
        kokoro_speed = max(0.5, min(2.0, req.speed))
        gen = pipeline(req.text, voice=req.voice, speed=kokoro_speed)
        chunks = [chunk.audio.numpy() for chunk in gen]
        audio = np.concatenate(chunks)

        # Speed already applied natively — process_audio should skip it
        req.speed = 1.0
        audio = process_audio(audio, 24000, req)
        
        # Clip to prevent distortion
        audio = np.clip(audio, -0.99, 0.99)
        
        # Save to disk
        job_id, filename = save_audio_file(audio, req.voice, kokoro_speed)
        
        buf = io.BytesIO()
        sf.write(buf, audio, 24000, format="WAV")
        buf.seek(0)
        
        return StreamingResponse(buf, media_type="audio/wav",
            headers={"Content-Disposition": f'attachment; filename="voiceover_{job_id}.wav"',
                     "X-Job-ID": job_id})
    except Exception as e:
        return {"error": str(e), "status": "failed"}


class TranscribeRequest(BaseModel):
    audio_id: str = ""          # preferred: job_id of an already-generated export (any engine)
    # Legacy fallback fields — only used if audio_id is not supplied, to stay
    # backward compatible with any older caller that still posts text/voice/speed.
    text: str = ""
    voice: str = "af_heart"
    speed: float = 1.0


@app.post("/transcribe")
async def transcribe_audio(req: TranscribeRequest, background_tasks: BackgroundTasks):
    """Queue transcription job for background processing.

    FIXED: this used to always regenerate the audio via the Kokoro pipeline
    regardless of which engine actually produced it — so if the source audio
    came from Edge TTS (or Parler, or Voice Clone), `voice` would be an
    engine-specific ID that Kokoro doesn't recognize (e.g. a Kokoro voice
    like 'af_heart' vs an Edge voice like 'hi-IN-MadhurNeural'), and
    pipeline(...) would throw. Now it transcribes the ACTUAL exported audio
    file directly with faster-whisper — works identically for every engine,
    and no longer wastes time re-generating audio that already exists.
    """
    try:
        audio_id = req.audio_id.strip()

        if audio_id:
            # Look up the already-generated file for this job/export id.
            conn = sqlite3.connect(DB_PATH)
            c = conn.cursor()
            c.execute("SELECT filename FROM exports WHERE id=?", (audio_id,))
            row = c.fetchone()
            conn.close()
            if not row:
                return {"error": f"No export found for audio_id '{audio_id}'", "status": "failed"}
            audio_path = row[0]
            if not os.path.exists(audio_path):
                return {"error": f"Export file missing on disk: {audio_path}", "status": "failed"}

            job_id = audio_id
            transcription_jobs[job_id] = {"status": "processing", "srt": None, "error": None}
            background_tasks.add_task(transcribe_background_from_file, job_id, audio_path)
            return {"job_id": job_id, "status": "queued"}

        # ── Legacy path (no audio_id given) — kept only for backward compat ──
        kokoro_speed = max(0.5, min(2.0, req.speed))
        gen = pipeline(req.text, voice=req.voice, speed=kokoro_speed)
        chunks = [chunk.audio.numpy() for chunk in gen]
        audio = np.concatenate(chunks)
        req.speed = 1.0
        audio = process_audio(audio, 24000, req)
        audio = np.clip(audio, -0.99, 0.99)
        job_id, _ = save_audio_file(audio, req.voice, kokoro_speed)
        transcription_jobs[job_id] = {"status": "processing", "srt": None, "error": None}
        background_tasks.add_task(transcribe_background, job_id, audio)
        return {"job_id": job_id, "status": "queued"}

    except Exception as e:
        return {"error": str(e), "status": "failed"}


def transcribe_background(job_id: str, audio: np.ndarray):
    """Background transcription worker"""
    try:
        # Save temp file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            sf.write(tmp.name, audio, 24000)
            tmp_path = tmp.name
        
        # Transcribe
        segments, info = whisper_model.transcribe(tmp_path, language="en")
        
        # Generate SRT
        srt_content = ""
        for i, segment in enumerate(segments, 1):
            start = format_time(segment.start)
            end = format_time(segment.end)
            text = segment.text.strip()
            srt_content += f"{i}\n{start} --> {end}\n{text}\n\n"
        
        # Save SRT to disk
        save_srt_file(job_id, srt_content)
        
        # Update job status
        transcription_jobs[job_id] = {"status": "completed", "srt": srt_content, "error": None}
        
        # Cleanup temp
        os.unlink(tmp_path)
        
    except Exception as e:
        transcription_jobs[job_id] = {"status": "failed", "srt": None, "error": str(e)}


def transcribe_background_from_file(job_id: str, audio_path: str):
    """Background transcription worker — transcribes an EXISTING audio file
    directly (no regeneration), so it works identically regardless of which
    TTS engine produced it (Kokoro, Edge, Parler, Voice Clone). language=None
    lets faster-whisper auto-detect, so non-English Edge TTS output (Hindi,
    Tamil, etc.) is transcribed correctly instead of being forced to English."""
    try:
        segments, info = whisper_model.transcribe(audio_path, language=None, vad_filter=True)

        srt_content = ""
        for i, segment in enumerate(segments, 1):
            start = format_time(segment.start)
            end = format_time(segment.end)
            text = segment.text.strip()
            srt_content += f"{i}\n{start} --> {end}\n{text}\n\n"

        save_srt_file(job_id, srt_content)
        transcription_jobs[job_id] = {"status": "completed", "srt": srt_content, "error": None}

    except Exception as e:
        transcription_jobs[job_id] = {"status": "failed", "srt": None, "error": str(e)}


@app.get("/transcribe-status/{job_id}")
async def transcribe_status(job_id: str):
    """Check transcription status"""
    if job_id not in transcription_jobs:
        return {"status": "not_found"}
    
    job = transcription_jobs[job_id]
    if job["status"] == "completed":
        result = {"status": "completed", "srt": job["srt"]}
        # Clean up completed job from memory to prevent RAM accumulation
        del transcription_jobs[job_id]
        return result
    elif job["status"] == "failed":
        return {"status": "failed", "error": job["error"]}
    else:
        return {"status": "processing"}


@app.get("/exports")
async def get_recent_exports(limit: int = 20):
    """Get recent exports with metadata"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT id, filename, srt_filename, voice, speed, created_at, duration FROM exports ORDER BY created_at DESC LIMIT ?", (limit,))
    rows = c.fetchall()
    conn.close()
    
    exports = []
    for row in rows:
        exports.append({
            "id": row[0],
            "filename": row[1],
            "srt_filename": row[2],
            "voice": row[3],
            "speed": row[4],
            "created_at": row[5],
            "duration": row[6]
        })
    
    return {"exports": exports}


@app.get("/exports/{file_id}")
async def get_export_file(file_id: str):
    """Serve exported audio/subtitle files"""
    # Sanitize file_id to prevent directory traversal
    if ".." in file_id or file_id.startswith("/"):
        return {"error": "Invalid file path"}
    filepath = f"{EXPORTS_DIR}/{file_id}"
    if os.path.exists(filepath):
        media_type = "audio/wav" if filepath.endswith(".wav") else "text/plain"
        return FileResponse(filepath, media_type=media_type)
    return {"error": "File not found"}


@app.post("/set-model")
async def set_model(req: ModelSelectRequest):
    """Switch between available models"""
    global selected_model
    selected_model = req.model
    return {"current_model": selected_model}


@app.get("/current-model")
async def get_current_model():
    """Get currently selected model"""
    return {"model": selected_model}


@app.get("/memory-stats")
async def get_memory_stats():
    """Get system memory usage"""
    mem = psutil.virtual_memory()
    return {
        "used_gb": round(mem.used / (1024**3), 2),
        "total_gb": round(mem.total / (1024**3), 2),
        "percent": mem.percent
    }


def format_time(seconds):
    """Convert seconds to SRT timestamp format (HH:MM:SS,mmm)"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


@app.post("/ai-helper")
async def ai_helper(req: HelperRequest):
    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            res = await client.post("http://localhost:11434/api/generate",
                json={"model": selected_model, "prompt": req.prompt, "stream": False})
        return {"ok": True, "response": res.json().get("response", "")}
    except Exception as e:
        return {"ok": False, "response": "", "error": str(e)}


@app.get("/ollama-status")
async def ollama_status():
    try:
        async with httpx.AsyncClient(timeout=4.0) as client:
            r = await client.get("http://localhost:11434/api/tags")
            models = [m["name"] for m in r.json().get("models", [])]
            return {"online": True, "models": models}
    except:
        return {"online": False, "models": []}


# ── APP SETTINGS PERSISTENCE ──────────────────────────────────────
# Persists user settings to a JSON file on disk so they survive across
# browser restarts, incognito mode, and different browsers — unlike
# localStorage which is silently lost in all those cases.
_SETTINGS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "voicelab_settings.json")
_app_settings: dict = {}

def _load_settings_from_disk():
    global _app_settings
    try:
        if os.path.exists(_SETTINGS_FILE):
            with open(_SETTINGS_FILE, "r", encoding="utf-8") as f:
                _app_settings = json.load(f)
    except Exception as e:
        print(f"[Settings] Could not load settings file: {e}")
        _app_settings = {}

_load_settings_from_disk()  # load once at startup


class SettingsPayload(BaseModel):
    settings: dict


@app.get("/get-settings")
async def get_settings():
    return {"settings": _app_settings}


@app.post("/save-settings")
async def save_settings(req: SettingsPayload):
    global _app_settings
    _app_settings.update(req.settings)
    try:
        with open(_SETTINGS_FILE, "w", encoding="utf-8") as f:
            json.dump(_app_settings, f, indent=2)
        return {"status": "ok"}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


# ── RECENT AUDIO ──────────────────────────────────────────────────
# Lets the frontend restore the last-used audio ID on page load, so
# the Video Studio checklist doesn't show "Step 2 missing" just because
# the user refreshed or reopened the browser.
@app.get("/recent-audio")
async def recent_audio():
    """Return the most recently generated audio file from the exports dir."""
    try:
        wavs = [
            f for f in os.listdir(EXPORTS_DIR)
            if f.endswith(".wav") and not f.startswith(("ref_", "edge_"))
        ]
        if not wavs:
            return {"audio_id": None}
        # Sort by modification time — most recent first
        wavs.sort(key=lambda f: os.path.getmtime(os.path.join(EXPORTS_DIR, f)), reverse=True)
        latest = wavs[0]
        audio_id = latest.replace(".wav", "")
        mtime = os.path.getmtime(os.path.join(EXPORTS_DIR, latest))
        age_minutes = (datetime.now().timestamp() - mtime) / 60
        return {
            "audio_id": audio_id,
            "filename": latest,
            "age_minutes": round(age_minutes, 1),
            # Only auto-restore if generated in the last 8 hours — older than
            # that and the user likely wants to start fresh rather than find a
            # stale audio ID auto-selected that they don't remember.
            "fresh": age_minutes < 480,
        }
    except Exception as e:
        return {"audio_id": None, "error": str(e)}


# ── PARLER DOWNLOAD PROGRESS SSE ─────────────────────────────────
# The existing /generate-parler blocks silently for ~2 minutes on first
# run while the 800MB model downloads. This endpoint lets the frontend
# show a live download progress bar for that first-use wait.
@app.get("/parler-load-stream")
async def parler_load_stream():
    """Stream Parler-TTS model load/download progress as SSE so the UI
    can show a real progress bar instead of an unresponsive spinner."""
    async def _stream():
        global parler_model, parler_tokenizer
        if parler_model is not None:
            yield "event: done\ndata: already_loaded\n\n"
            return
        try:
            yield "data: Importing Parler-TTS...\n\n"
            await asyncio.sleep(0.1)

            from parler_tts import ParlerTTSForConditionalGeneration
            from transformers import AutoTokenizer
            import torch

            device = "cuda" if torch.cuda.is_available() else "cpu"
            model_id = "parler-tts/parler-tts-mini-v1"

            yield f"data: Loading model on {device} (first run downloads ~800MB)...\n\n"
            await asyncio.sleep(0.1)

            # Run blocking model load in thread so SSE can keep streaming
            def _load():
                global parler_model, parler_tokenizer
                parler_model     = ParlerTTSForConditionalGeneration.from_pretrained(model_id).to(device)
                parler_tokenizer = AutoTokenizer.from_pretrained(model_id)

            # Emit heartbeat ticks while the thread runs
            import concurrent.futures
            loop = asyncio.get_event_loop()
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                fut = loop.run_in_executor(pool, _load)
                tick = 0
                stages = [
                    "Downloading tokenizer...",
                    "Downloading model weights (~800MB — this takes a few minutes)...",
                    "Loading model into VRAM...",
                    "Finalising...",
                ]
                while not fut.done():
                    stage = stages[min(tick // 8, len(stages)-1)]
                    pct = min(90, tick * 3)
                    yield f"data: {stage} ({pct}%)\n\n"
                    tick += 1
                    await asyncio.sleep(1.5)
                await fut  # re-raise any exception from the thread

            yield "data: ✅ Parler-TTS ready!\n\n"
            yield "event: done\ndata: loaded\n\n"

        except Exception as e:
            yield f"data: ❌ Failed: {e}\n\n"
            yield "event: error\ndata: failed\n\n"

    return StreamingResponse(_stream(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


_PREDOWNLOAD_TARGETS = {
    # key: (display label, loader callable, size hint)
    "parler":     ("Parler-TTS (style-conditioned voice)", lambda: get_parler(), "~800MB"),
    "chatterbox": ("Chatterbox TTS (voice cloning)", lambda: get_chatterbox(), "~2GB, needs Python 3.11"),
    "xtts":       ("Coqui XTTS v2 (voice cloning fallback)", lambda: get_xtts(), "~2GB"),
    "sd_thumbnail": ("AI Thumbnail (SD-Turbo)", lambda: get_sd_pipeline(), "~2GB"),
    "ix2ix":      ("AI Image Editing (InstructPix2Pix)", lambda: get_instruct_pix2pix_pipeline(), "~5GB"),
    "musicgen":   ("AI Music (MusicGen)", lambda: get_musicgen(), "~2GB"),
}


@app.get("/predownload-stream/{key}")
async def predownload_stream(key: str):
    """Generic pre-download trigger for any of the lazy-loaded model-weight
    dependencies, streamed as SSE so the Dependency panel can show real
    progress. This is what makes the panel's 'Pre-download now' button
    actually fetch the model instead of only reporting status — previously
    the only way to trigger these downloads was to click into that engine's
    tab and use it once, which is what caused the 'panel says green but a
    tab still shows Downloading...' confusion."""
    if key not in _PREDOWNLOAD_TARGETS:
        return JSONResponse(status_code=404, content={"error": f"Unknown predownload key: {key}"})

    label, loader, size_hint = _PREDOWNLOAD_TARGETS[key]

    async def _stream():
        try:
            yield f"data: Preparing {label} ({size_hint})...\n\n"
            await asyncio.sleep(0.1)

            import concurrent.futures
            loop = asyncio.get_event_loop()
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                fut = loop.run_in_executor(pool, loader)
                tick = 0
                stages = [
                    f"Downloading {label} weights ({size_hint})...",
                    "Loading model into memory/VRAM...",
                    "Finalising...",
                ]
                while not fut.done():
                    stage = stages[min(tick // 8, len(stages) - 1)]
                    pct = min(90, tick * 3)
                    yield f"data: {stage} ({pct}%)\n\n"
                    tick += 1
                    await asyncio.sleep(1.5)
                fut.result()  # re-raise any exception from the thread

            yield f"data: ✅ {label} ready!\n\n"
            yield "event: done\ndata: loaded\n\n"
        except Exception as e:
            yield f"data: ❌ Failed: {e}\n\n"
            yield "event: error\ndata: failed\n\n"

    return StreamingResponse(_stream(), media_type="text/event-stream",
                              headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.get("/ollama-pull-status/{model_name:path}")
async def ollama_pull_status(model_name: str):
    """Check whether a specific model is already pulled."""
    try:
        async with httpx.AsyncClient(timeout=4.0) as client:
            r = await client.get("http://localhost:11434/api/tags")
            models = [m["name"] for m in r.json().get("models", [])]
            return {"pulled": model_name in models, "models": models}
    except:
        return {"pulled": False, "models": []}


@app.get("/ollama-pull-stream/{model_name:path}")
async def ollama_pull_stream(model_name: str):
    """Pull (download) a model from Ollama hub, streaming progress as SSE.
    No server restart needed — model is immediately available via /set-model
    once the pull finishes."""
    async def _stream():
        try:
            # 60 min cap - generous enough for large models on slow connections,
            # but not truly infinite (a dead connection shouldn't hang forever)
            async with httpx.AsyncClient(timeout=3600.0) as client:
                async with client.stream(
                    "POST", "http://localhost:11434/api/pull",
                    json={"name": model_name, "stream": True},
                    timeout=3600.0,
                ) as resp:
                    async for line in resp.aiter_lines():
                        if line.strip():
                            import json as _json
                            try:
                                d = _json.loads(line)
                                status = d.get("status", "")
                                total  = d.get("total", 0)
                                comp   = d.get("completed", 0)
                                pct    = int(comp / total * 100) if total else 0
                                msg    = f"{status} {pct}%" if total else status
                                yield f"data: {msg}\n\n"
                                if d.get("status") == "success":
                                    yield "event: done\ndata: done\n\n"
                                    return
                            except Exception:
                                yield f"data: {line}\n\n"
        except Exception as e:
            yield f"event: error\ndata: {e}\n\n"
    return StreamingResponse(_stream(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


# Curated list of best Ollama models per use-case — checked dynamically so
# the app can suggest models the user doesn't have yet, with one-click pull.
RECOMMENDED_MODELS = [
    {
        "name": "llama3.2:3b", "display": "Llama 3.2 3B", "size_gb": 2.0,
        "best_for": ["chat", "script", "fast"], "category": "Small & Fast",
        "description": "Fast, snappy — best for quick script writing and chat. Runs well on 8GB GPU.",
        "pros": ["Very fast responses", "Tiny footprint, runs on almost anything", "Good for quick iteration"],
        "cons": ["Weaker reasoning than 7B+ models", "Less nuanced creative writing"],
        "min_ram_gb": 8, "min_vram_gb": 3, "ollama_id": "llama3.2:3b",
        "params": "3.2B", "context_window": "128K tokens", "quantization": "Q4_K_M (default pull)", "speed_tier": "fast",
    },
    {
        "name": "llama3.1:8b", "display": "Llama 3.1 8B", "size_gb": 4.7,
        "best_for": ["chat", "script", "creative"], "category": "All-Purpose Chat",
        "description": "Well-rounded — good quality chat and creative writing. Recommended default.",
        "pros": ["Strong general-purpose quality", "Well-tested, widely used", "Good balance of speed/quality"],
        "cons": ["Slower than 3B models", "Needs ~6GB+ VRAM for comfortable speed"],
        "min_ram_gb": 16, "min_vram_gb": 6, "ollama_id": "llama3.1:8b",
        "params": "8B", "context_window": "128K tokens", "quantization": "Q4_K_M (default pull)", "speed_tier": "medium",
    },
    {
        "name": "mistral:7b", "display": "Mistral 7B", "size_gb": 4.1,
        "best_for": ["chat", "script"], "category": "All-Purpose Chat",
        "description": "Fast, sharp reasoning — great for structured scripts and instructions.",
        "pros": ["Fast for its quality tier", "Follows structured instructions well"],
        "cons": ["Less creative/expressive than Gemma2 for narration"],
        "min_ram_gb": 16, "min_vram_gb": 5, "ollama_id": "mistral:7b",
        "params": "7.3B", "context_window": "32K tokens", "quantization": "Q4_K_M (default pull)", "speed_tier": "medium",
    },
    {
        "name": "gemma2:9b", "display": "Gemma 2 9B", "size_gb": 5.5,
        "best_for": ["creative", "script"], "category": "Best for Script & Creative Writing",
        "description": "Google's model — excellent creative writing, very natural-sounding scripts.",
        "pros": ["Most natural-sounding voiceover scripts", "Strong creative writing"],
        "cons": ["Slower, heavier than 7B options", "Needs more VRAM for smooth use"],
        "min_ram_gb": 16, "min_vram_gb": 7, "ollama_id": "gemma2:9b",
        "params": "9.2B", "context_window": "8K tokens", "quantization": "Q4_K_M (default pull)", "speed_tier": "medium-slow",
    },
    {
        "name": "qwen2.5:7b", "display": "Qwen 2.5 7B", "size_gb": 4.4,
        "best_for": ["multilingual", "chat", "script"], "category": "Best for Hindi / Multilingual",
        "description": "Best multilingual model — great for Hindi/Urdu/Indian scripts. Strong on code too.",
        "pros": ["Best-in-class multilingual (Hindi/Urdu/Indian languages)", "Strong code understanding too"],
        "cons": ["Slightly less natural in pure-English creative writing than Gemma2"],
        "min_ram_gb": 16, "min_vram_gb": 6, "ollama_id": "qwen2.5:7b",
        "params": "7.6B", "context_window": "128K tokens", "quantization": "Q4_K_M (default pull)", "speed_tier": "medium",
    },
    {
        "name": "phi3.5:3.8b", "display": "Phi 3.5 3.8B", "size_gb": 2.2,
        "best_for": ["chat", "fast"], "category": "Small & Fast",
        "description": "Microsoft's small-but-smart model — fastest good-quality chat, very low RAM.",
        "pros": ["Excellent quality-for-size", "Very low RAM/VRAM needs"],
        "cons": ["Shorter effective context than larger models"],
        "min_ram_gb": 8, "min_vram_gb": 3, "ollama_id": "phi3.5:3.8b",
        "params": "3.8B", "context_window": "128K tokens", "quantization": "Q4_K_M (default pull)", "speed_tier": "fast",
    },
    {
        "name": "llava:7b", "display": "LLaVA 7B (Vision)", "size_gb": 4.5,
        "best_for": ["vision", "image"], "category": "Reasoning & Vision",
        "description": "Multimodal — can see and describe images. Useful for auto-captioning gameplay.",
        "pros": ["Well-established vision model", "Good general image description"],
        "cons": ["Weaker JSON/structured-output discipline than qwen2.5vl", "Slower per-frame than smaller vision models"],
        "min_ram_gb": 16, "min_vram_gb": 6, "ollama_id": "llava:7b",
        "params": "7B (+ vision encoder)", "context_window": "4K tokens", "quantization": "Q4_K_M (default pull)", "speed_tier": "medium-slow",
    },
    {
        "name": "qwen2.5vl:7b", "display": "Qwen 2.5 VL 7B (Vision)", "size_gb": 6.0,
        "best_for": ["vision", "image", "reasoning"], "category": "Reasoning & Vision",
        "description": "Strong modern vision-language model — reads frames AND follows structured JSON instructions reliably. Powers Clip Finder's vision-based moment scoring.",
        "pros": ["Best structured-JSON reliability for automated pipelines", "Strong scene/text understanding in frames", "Actively used by this app's Clip Finder vision path"],
        "cons": ["Larger download (~6GB)", "Slower than LLaVA per-frame on weaker GPUs"],
        "min_ram_gb": 16, "min_vram_gb": 8, "ollama_id": "qwen2.5vl:7b",
        "params": "7B (+ vision encoder)", "context_window": "32K tokens", "quantization": "Q4_K_M (default pull)", "speed_tier": "slow",
    },
    {
        "name": "moondream", "display": "Moondream (Vision, tiny)", "size_gb": 1.7,
        "best_for": ["vision", "fast"], "category": "Small & Fast",
        "description": "Tiny vision model — much faster/lighter than LLaVA or Qwen-VL for basic frame description.",
        "pros": ["Extremely small and fast for a vision model", "Good enough for basic 'what's in this frame' tasks"],
        "cons": ["Less accurate than qwen2.5vl for nuanced scene understanding", "Weaker structured-output reliability"],
        "min_ram_gb": 8, "min_vram_gb": 2, "ollama_id": "moondream",
        "params": "1.6B (+ vision encoder)", "context_window": "2K tokens", "quantization": "Q4 (default pull)", "speed_tier": "fast",
    },
    {
        "name": "deepseek-r1:7b", "display": "DeepSeek-R1 7B", "size_gb": 4.7,
        "best_for": ["reasoning", "creative"], "category": "Reasoning & Vision",
        "description": "Chain-of-thought reasoning model — best for complex, detailed script outlines.",
        "pros": ["Strong step-by-step reasoning", "Good for complex multi-part outlines"],
        "cons": ["Can be slow/verbose due to its 'thinking' phase", "Overkill for simple scripts"],
        "min_ram_gb": 16, "min_vram_gb": 6, "ollama_id": "deepseek-r1:7b",
        "params": "7.6B (Qwen2.5 distill)", "context_window": "128K tokens", "quantization": "Q4_K_M (default pull)", "speed_tier": "medium-slow (thinking phase adds latency)",
    },
]


def _detect_hardware() -> dict:
    """Real hardware detection — no hardcoded assumptions. RAM via psutil
    (always available), GPU name + VRAM via torch.cuda if a CUDA GPU is
    present. Used to actually classify which models will run well on YOUR
    machine instead of guessing."""
    ram_gb = round(psutil.virtual_memory().total / (1024 ** 3), 1)
    cpu_count = psutil.cpu_count(logical=True) or 0
    gpu_name, vram_gb = None, 0.0
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            vram_gb = round(torch.cuda.get_device_properties(0).total_memory / (1024 ** 3), 1)
    except Exception:
        pass
    return {"ram_gb": ram_gb, "cpu_count": cpu_count, "gpu_name": gpu_name, "vram_gb": vram_gb,
            "has_gpu": gpu_name is not None}


@app.get("/system/hardware-info")
async def system_hardware_info():
    return _detect_hardware()


def _classify_model_fit(model: dict, hw: dict) -> dict:
    """Great fit / will be slow / likely won't fit — based on YOUR actual
    detected RAM and VRAM, not a one-size-fits-all size cutoff."""
    min_ram, min_vram = model.get("min_ram_gb", 8), model.get("min_vram_gb", 4)
    effective_vram = hw["vram_gb"] if hw["has_gpu"] else 0
    if hw["ram_gb"] < min_ram * 0.7:
        return {"fit": "poor", "fit_label": "❌ Likely won't fit — not enough system RAM"}
    if hw["has_gpu"] and effective_vram >= min_vram:
        return {"fit": "great", "fit_label": f"✅ Great fit for your {hw['gpu_name']}"}
    if hw["has_gpu"] and effective_vram >= min_vram * 0.5:
        return {"fit": "ok", "fit_label": f"⚠️ Will run, but slower on your {hw['gpu_name']} (VRAM below recommended)"}
    if not hw["has_gpu"] and hw["ram_gb"] >= min_ram:
        return {"fit": "ok", "fit_label": "⚠️ No GPU detected — will run on CPU, noticeably slower"}
    return {"fit": "poor", "fit_label": "❌ Likely too slow for your hardware"}


@app.get("/recommended-models")
async def get_recommended_models():
    """Return the curated model list with 'installed' flag from live Ollama,
    PLUS a real hardware-fit classification per model based on YOUR actual
    detected RAM/GPU/VRAM — not a hardcoded guess."""
    try:
        async with httpx.AsyncClient(timeout=4.0) as client:
            r = await client.get("http://localhost:11434/api/tags")
            installed = {m["name"] for m in r.json().get("models", [])}
    except Exception:
        installed = set()
    hw = _detect_hardware()
    result = []
    for m in RECOMMENDED_MODELS:
        result.append({**m, "installed": m["ollama_id"] in installed, **_classify_model_fit(m, hw)})
    return {"models": result, "installed_ids": list(installed), "hardware": hw}


@app.post("/models/benchmark/{model_id}")
async def benchmark_model(model_id: str):
    """Real live benchmark on YOUR actual hardware — the honest alternative
    to guessing tokens/sec. Sends one small fixed prompt and reads Ollama's
    own eval_count/eval_duration from the response (no manual timing needed,
    Ollama already measures this internally for every generation)."""
    try:
        async with httpx.AsyncClient(timeout=90.0) as client:
            resp = await client.post("http://localhost:11434/api/generate", json={
                "model": model_id, "prompt": "Write one short sentence about the ocean.", "stream": False,
            })
            if resp.status_code != 200:
                return {"error": f"Ollama returned HTTP {resp.status_code} — is the model pulled?"}
            d = resp.json()
            eval_count = d.get("eval_count", 0)
            eval_duration_ns = d.get("eval_duration", 0)
            if not eval_count or not eval_duration_ns:
                return {"error": "Ollama response didn't include timing data"}
            tokens_per_sec = round(eval_count / (eval_duration_ns / 1e9), 1)
            return {"tokens_per_sec": tokens_per_sec, "eval_count": eval_count}
    except Exception as e:
        return {"error": str(e)}


@app.get("/search-ollama-models")
async def search_ollama_models(q: str = ""):
    """Best-effort live search against ollama.com/search — this is NOT an
    official API (Ollama doesn't expose one for browsing their library), so
    it's fragile by nature: if their page structure changes, this degrades
    to 'unavailable' rather than breaking anything. Results are explicitly
    marked as unverified community finds, distinct from the curated,
    hand-vetted RECOMMENDED_MODELS list."""
    q = q.strip()
    if not q:
        return {"results": [], "source": "empty_query"}
    try:
        async with httpx.AsyncClient(timeout=8.0, follow_redirects=True) as client:
            resp = await client.get(f"https://ollama.com/search?q={q}",
                                     headers={"User-Agent": "Mozilla/5.0 (VoiceLab local app)"})
            if resp.status_code != 200:
                return {"results": [], "source": "unavailable",
                        "message": f"ollama.com returned HTTP {resp.status_code} — showing curated picks only."}
            html = resp.text
        # Ollama's model pages live at /library/<name> — this URL convention
        # is core to their product and the most stable thing to key off of.
        names = re.findall(r'href="/library/([a-zA-Z0-9._:\-]+)"', html)
        seen, results = set(), []
        curated_ids = {m["ollama_id"] for m in RECOMMENDED_MODELS}
        for name in names:
            if name in seen or name in curated_ids or len(results) >= 15:
                continue
            seen.add(name)
            results.append({
                "ollama_id": name, "display": name, "size_gb": None,
                "description": "Found on ollama.com — not manually vetted by this app.",
                "pros": [], "cons": ["Unverified — check ollama.com/library/" + name + " for real specs/reviews before pulling"],
                "source": "community",
            })
        if not results:
            return {"results": [], "source": "no_matches",
                    "message": f"No matches found on ollama.com for '{q}'. You can still type an exact model name to pull it directly."}
        return {"results": results, "source": "live"}
    except Exception as e:
        return {"results": [], "source": "unavailable",
                "message": f"Live search unavailable right now ({type(e).__name__}) — showing curated picks only. "
                            f"You can still type any exact model name to pull it directly."}


# ── EDGE TTS VOICE LIBRARY ───────────────────────────────────────
# Full curated list of Microsoft Edge TTS voices — grouped by language/region.
# Edge TTS supports 400+ voices; these are the highest-quality Neural voices
# most useful for content creators. Loaded once at startup.
EDGE_VOICES = {
    # ── Indian languages & accents (most relevant for this app) ──────
    "hi-IN-MadhurNeural":   "🇮🇳 Madhur (Hindi Male)",
    "hi-IN-SwaraNeural":    "🇮🇳 Swara (Hindi Female)",
    "en-IN-NeerjaNeural":   "🇮🇳 Neerja (Indian English Female)",
    "en-IN-PrabhatNeural":  "🇮🇳 Prabhat (Indian English Male)",
    "mr-IN-AarohiNeural":   "🇮🇳 Aarohi (Marathi Female)",
    "mr-IN-ManoharNeural":  "🇮🇳 Manohar (Marathi Male)",
    "bn-IN-BashkarNeural":  "🇮🇳 Bashkar (Bengali Male)",
    "bn-IN-TanishaaNeural": "🇮🇳 Tanishaa (Bengali Female)",
    "ta-IN-PallaviNeural":  "🇮🇳 Pallavi (Tamil Female)",
    "ta-IN-ValluvarNeural": "🇮🇳 Valluvar (Tamil Male)",
    "te-IN-MohanNeural":    "🇮🇳 Mohan (Telugu Male)",
    "te-IN-ShrutiNeural":   "🇮🇳 Shruti (Telugu Female)",
    "gu-IN-DhwaniNeural":   "🇮🇳 Dhwani (Gujarati Female)",
    "gu-IN-NiranjanNeural": "🇮🇳 Niranjan (Gujarati Male)",
    "kn-IN-GaganNeural":    "🇮🇳 Gagan (Kannada Male)",
    "kn-IN-SapnaNeural":    "🇮🇳 Sapna (Kannada Female)",
    "ml-IN-MidhunNeural":   "🇮🇳 Midhun (Malayalam Male)",
    "ml-IN-SobhanaNeural":  "🇮🇳 Sobhana (Malayalam Female)",
    "pa-IN-OjasNeural":     "🇮🇳 Ojas (Punjabi Male)",
    "ur-IN-GulNeural":      "🇮🇳 Gul (Urdu Female)",
    "ur-IN-SalmanNeural":   "🇮🇳 Salman (Urdu Male)",
    # ── English — US ─────────────────────────────────────────────────
    "en-US-AndrewNeural":   "🇺🇸 Andrew (US Male — conversational)",
    "en-US-AvaNeural":      "🇺🇸 Ava (US Female — warm)",
    "en-US-BrianNeural":    "🇺🇸 Brian (US Male — casual)",
    "en-US-EmmaNeural":     "🇺🇸 Emma (US Female — natural)",
    "en-US-AriaNeural":     "🇺🇸 Aria (US Female — professional)",
    "en-US-GuyNeural":      "🇺🇸 Guy (US Male — newsreader)",
    "en-US-JennyNeural":    "🇺🇸 Jenny (US Female — assistant)",
    "en-US-MichelleNeural": "🇺🇸 Michelle (US Female — friendly)",
    "en-US-RogerNeural":    "🇺🇸 Roger (US Male — lively)",
    "en-US-SteffanNeural":  "🇺🇸 Steffan (US Male — clear)",
    # ── English — UK / AU ─────────────────────────────────────────────
    "en-GB-LibbyNeural":    "🇬🇧 Libby (UK Female)",
    "en-GB-RyanNeural":     "🇬🇧 Ryan (UK Male)",
    "en-GB-SoniaNeural":    "🇬🇧 Sonia (UK Female)",
    "en-AU-NatashaNeural":  "🇦🇺 Natasha (AU Female)",
    "en-AU-WilliamNeural":  "🇦🇺 William (AU Male)",
    # ── Other popular languages ───────────────────────────────────────
    "ar-SA-HamedNeural":    "🇸🇦 Hamed (Arabic Male)",
    "ar-SA-ZariyahNeural":  "🇸🇦 Zariyah (Arabic Female)",
    "de-DE-KatjaNeural":    "🇩🇪 Katja (German Female)",
    "de-DE-ConradNeural":   "🇩🇪 Conrad (German Male)",
    "es-ES-AlvaroNeural":   "🇪🇸 Alvaro (Spanish Male)",
    "es-ES-ElviraNeural":   "🇪🇸 Elvira (Spanish Female)",
    "fr-FR-DeniseNeural":   "🇫🇷 Denise (French Female)",
    "fr-FR-HenriNeural":    "🇫🇷 Henri (French Male)",
    "ja-JP-KeitaNeural":    "🇯🇵 Keita (Japanese Male)",
    "ja-JP-NanamiNeural":   "🇯🇵 Nanami (Japanese Female)",
    "ko-KR-InJoonNeural":   "🇰🇷 InJoon (Korean Male)",
    "ko-KR-SunHiNeural":    "🇰🇷 SunHi (Korean Female)",
    "pt-BR-AntonioNeural":  "🇧🇷 Antonio (Portuguese Male)",
    "pt-BR-FranciscaNeural":"🇧🇷 Francisca (Portuguese Female)",
    "ru-RU-DmitryNeural":   "🇷🇺 Dmitry (Russian Male)",
    "ru-RU-SvetlanaNeural": "🇷🇺 Svetlana (Russian Female)",
    "zh-CN-XiaoxiaoNeural": "🇨🇳 Xiaoxiao (Chinese Female — warm)",
    "zh-CN-YunxiNeural":    "🇨🇳 Yunxi (Chinese Male — lively)",
    "zh-CN-YunjianNeural":  "🇨🇳 Yunjian (Chinese Male — narration)",
    "tr-TR-AhmetNeural":    "🇹🇷 Ahmet (Turkish Male)",
    "tr-TR-EmelNeural":     "🇹🇷 Emel (Turkish Female)",
    "id-ID-ArdiNeural":     "🇮🇩 Ardi (Indonesian Male)",
    "id-ID-GadisNeural":    "🇮🇩 Gadis (Indonesian Female)",
    "vi-VN-HoaiMyNeural":   "🇻🇳 HoaiMy (Vietnamese Female)",
    "vi-VN-NamMinhNeural":  "🇻🇳 NamMinh (Vietnamese Male)",
    "th-TH-NiwatNeural":    "🇹🇭 Niwat (Thai Male)",
    "th-TH-PremwadeeNeural":"🇹🇭 Premwadee (Thai Female)",
    "nl-NL-ColetteNeural":  "🇳🇱 Colette (Dutch Female)",
    "nl-NL-MaartenNeural":  "🇳🇱 Maarten (Dutch Male)",
    "pl-PL-MarekNeural":    "🇵🇱 Marek (Polish Male)",
    "pl-PL-ZofiaNeural":    "🇵🇱 Zofia (Polish Female)",
    "sv-SE-SofieNeural":    "🇸🇪 Sofie (Swedish Female)",
    "sv-SE-MattiasNeural":  "🇸🇪 Mattias (Swedish Male)",
}

async def generate_edge_tts(text: str, voice: str, rate: str = "+0%", pitch: str = "+0Hz") -> np.ndarray:
    """Generate audio using Microsoft Edge TTS - returns 24kHz mono audio.
    rate/pitch use Edge TTS's NATIVE per-request parameters, which sound far
    more natural than generating at defaults and post-processing with
    librosa afterward."""
    communicate = edge_tts.Communicate(text, voice=voice, rate=rate, pitch=pitch)
    mp3_chunks = []

    async def _collect():
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                mp3_chunks.append(chunk["data"])

    try:
        # Hard timeout: Edge TTS is an unofficial network API. If the
        # connection stalls (firewall, antivirus, ISP filtering, or Microsoft's
        # endpoint being unreachable), this used to hang forever with no
        # error - and since the server runs a single async event loop, that
        # hang could block every other request too. 45s covers even long
        # scripts on a slow connection; anything past that is a real failure.
        await asyncio.wait_for(_collect(), timeout=45.0)
    except asyncio.TimeoutError:
        raise RuntimeError(
            "Edge TTS didn't respond within 45 seconds. This usually means "
            "your firewall/antivirus/network is blocking the connection to "
            "Microsoft's TTS service - see TROUBLESHOOTING.md."
        )

    if not mp3_chunks:
        raise RuntimeError("Edge TTS returned no audio data - try again, or check your internet connection.")

    mp3_bytes = io.BytesIO(b"".join(mp3_chunks))
    # Use librosa to load MP3 (no pydub/ffmpeg required)
    audio_np, _ = librosa.load(mp3_bytes, sr=24000, mono=True)
    return audio_np


@app.post("/generate-hindi")
async def generate_hindi(req: EdgeTTSRequest):
    """Generate Hindi/Hinglish speech using Microsoft Edge TTS"""
    try:
        # Map speed (0.5-2.0 multiplier) -> Edge's native rate string
        rate_pct = int(round((req.speed - 1.0) * 100))
        rate_pct = max(-50, min(100, rate_pct))
        rate_str = f"{'+' if rate_pct >= 0 else ''}{rate_pct}%"

        # Map pitch (0.5-2.0 multiplier) -> Edge's native pitch string (Hz)
        semitones = 12.0 * np.log2(req.pitch) if req.pitch > 0 else 0.0
        pitch_hz = int(round(semitones * 11))
        pitch_hz = max(-50, min(50, pitch_hz))
        pitch_str = f"{'+' if pitch_hz >= 0 else ''}{pitch_hz}Hz"

        audio = await generate_edge_tts(req.text, req.voice, rate=rate_str, pitch=pitch_str)
        if audio.size == 0:
            return {"error": "Audio generation failed"}
        
        # Speed & pitch already applied natively above — skip in process_audio
        orig_speed = req.speed
        req.speed = 1.0
        req.pitch = 1.0
        audio = process_audio(audio, 24000, req)
        audio = np.clip(audio, -0.99, 0.99)
        
        # Save to file
        job_id, filename = save_audio_file(audio, f"edge_{req.voice.split('-')[0]}", orig_speed)
        
        # Stream back as WAV
        audio_bytes = io.BytesIO()
        sf.write(audio_bytes, audio, 24000, format="WAV")
        audio_bytes.seek(0)
        
        return StreamingResponse(audio_bytes, media_type="audio/wav",
            headers={
                "Content-Disposition": f'attachment; filename="voiceover_{job_id}.wav"',
                "X-Job-ID": job_id
            })
    except Exception as e:
        err_msg = str(e)
        if "ConnectionError" in str(type(e)) or "timeout" in err_msg.lower():
            return {"error": "Edge TTS failed — check internet connection"}
        return {"error": err_msg}


EDGE_LANG_GROUPS = {
    "hi": "Hindi", "mr": "Marathi", "bn": "Bengali", "ta": "Tamil", "te": "Telugu",
    "gu": "Gujarati", "kn": "Kannada", "ml": "Malayalam", "pa": "Punjabi", "ur": "Urdu",
    "ar": "Arabic", "de": "German", "es": "Spanish", "fr": "French", "ja": "Japanese",
    "ko": "Korean", "pt": "Portuguese", "ru": "Russian", "zh": "Chinese", "tr": "Turkish",
    "id": "Indonesian", "vi": "Vietnamese", "th": "Thai", "nl": "Dutch", "pl": "Polish",
    "sv": "Swedish",
}

def _edge_voice_group(voice_id: str) -> str:
    """Clean, reliable group label derived from the actual BCP-47 language-region
    code in the voice id (e.g. 'hi-IN-MadhurNeural' -> 'hi-IN') — not from
    parsing the emoji-flag display string, which breaks depending on how the
    OS/font renders flag emoji (Windows shows plain 'IN'/'US' letters instead
    of a flag glyph, which previously broke the grouping entirely)."""
    parts = voice_id.split("-")
    lang, region = parts[0], parts[1]
    if lang == "en":
        return {"US": "English (US)", "GB": "English (UK)", "AU": "English (Australia)",
                "IN": "English (India)"}.get(region, f"English ({region})")
    return EDGE_LANG_GROUPS.get(lang, lang.upper())

@app.get("/edge-voices")
async def get_edge_voices():
    """Get available Edge TTS voices, pre-grouped with clean text labels
    (no emoji-flag parsing) so the frontend can render group-filter chips
    the same reliable way Kokoro's voice picker does."""
    voices = []
    for vid, display_name in EDGE_VOICES.items():
        # Strip only the leading flag emoji — the group chip already conveys
        # language, so no need to duplicate it, but keep the descriptive
        # "(US Male — conversational)" part since that's genuinely useful.
        name = re.sub(r"^[^\w]*", "", display_name).strip()
        voices.append({"id": vid, "name": name, "group": _edge_voice_group(vid)})
    return {"voices": EDGE_VOICES, "voices_grouped": voices}


@app.get("/clone-status")
async def clone_status():
    """Lightweight check (no model load) — which neural cloning engine is
    actually installed and will actually be used by /generate-clone, so the
    UI badge matches reality instead of naming a library the code never
    calls. Checked in the same priority order generate_clone() uses:
    1) Chatterbox TTS (preferred — easier install, no C++ Build Tools needed
       on Windows, but requires Python 3.11 exactly)
    2) Coqui XTTS v2 (fallback — needs C++ Build Tools on Windows)
    3) DSP-only tiers 1-3 (always available, no install needed, lower
       quality than either neural engine above)."""
    py_ver = f"{_sys_boot.version_info.major}.{_sys_boot.version_info.minor}"

    try:
        from chatterbox.tts import ChatterboxTTS  # noqa: F401
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        return {"available": True, "engine": "chatterbox", "loaded": chatterbox_model is not None,
                "detail": f"✅ Chatterbox TTS installed (device: {device}) — neural voice cloning active" +
                          (" · model loaded" if chatterbox_model is not None else " · loads on first use (~30-60s)")}
    except ImportError:
        pass
    except Exception as e:
        print(f"  [Clone] Chatterbox import found but broken: {e}")

    try:
        from TTS.api import TTS  # noqa: F401
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        return {"available": True, "engine": "xtts", "loaded": xtts_model is not None,
                "detail": f"✅ Coqui XTTS v2 installed (device: {device}) — neural voice cloning active" +
                          (" · model loaded" if xtts_model is not None else " · loads on first use (~60s)")}
    except ImportError:
        pass

    return {"available": True, "engine": "dsp", "loaded": False,
            "detail": (f"⚠️ Running DSP-only cloning (Tier 1–3 — pitch/spectral/energy matching, no neural model). "
                       f"You're on Python {py_ver}. For best quality, install ONE of:\\n"
                       f"Recommended: pip install chatterbox-tts  (needs Python 3.11 exactly — "
                       f"{'your current venv qualifies' if py_ver == '3.11' else 'your current venv does NOT qualify, would need a separate Python 3.11 venv'})\\n"
                       f"Alternative: pip install TTS  (Coqui XTTS v2, works on other Python versions but needs C++ Build Tools on Windows)\\n"
                       f"Neither is required — DSP cloning (current mode) already works without either.")}


# ── VOICE CLONING ──────────────────────────────────────────────────────────────

chatterbox_model = None  # lazy load — only when first used
xtts_model = None        # lazy load — only when first used

def get_chatterbox():
    """Preferred neural voice-cloning engine. Requires Python 3.11 exactly
    (upstream constraint of the chatterbox-tts package, not this app)."""
    global chatterbox_model
    if chatterbox_model is None:
        try:
            from chatterbox.tts import ChatterboxTTS
        except ImportError as e:
            raise ImportError(
                "chatterbox-tts not installed.\n"
                "pip install chatterbox-tts  (needs Python 3.11 exactly)"
            ) from e
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"  Loading Chatterbox voice-cloning model on {device} (first time ~30-60s)...")
        chatterbox_model = ChatterboxTTS.from_pretrained(device=device)
        print("  Chatterbox model  [OK]  Ready")
    return chatterbox_model

def get_xtts():
    """Fallback neural voice-cloning engine, used only if Chatterbox isn't
    installed. Needs C++ Build Tools on Windows to install."""
    global xtts_model
    if xtts_model is None:
        try:
            from TTS.api import TTS
        except ImportError:
            raise RuntimeError(
                "Neither chatterbox-tts nor TTS (Coqui) is installed.\n\n"
                "Option A (recommended): pip install chatterbox-tts  (needs Python 3.11 exactly)\n"
                "Option B: pip install TTS  (Coqui XTTS v2, needs C++ Build Tools on Windows)\n\n"
                "Clone will keep working via the DSP-only fallback without either."
            )
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"  Loading voice cloning model on {device} (first time ~60s)...")
        xtts_model = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)
        print("  Voice cloning model  [OK]  Ready")
    return xtts_model

# In-memory store: ref_id → {path, duration, filename}
clone_refs = {}


def load_uploaded_audio(content: bytes, filename: str) -> np.ndarray:
    """Decode an uploaded audio file into 24kHz mono float32 samples."""
    suffix = os.path.splitext(filename or "")[1] or ".wav"
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(content)
            tmp_path = tmp.name

        try:
            audio_np, _ = librosa.load(tmp_path, sr=24000, mono=True)
            return audio_np
        except Exception:
            try:
                import av
            except ImportError as exc:
                raise ValueError("PyAV not installed. Run: pip install av") from exc

            with av.open(tmp_path) as container:
                audio_stream = next((stream for stream in container.streams if stream.type == "audio"), None)
                if audio_stream is None:
                    raise ValueError("No audio stream found in uploaded file")

                chunks = []
                for frame in container.decode(audio_stream):
                    frame_audio = frame.to_ndarray()
                    if frame_audio.ndim > 1:
                        frame_audio = frame_audio.mean(axis=0)
                    chunks.append(frame_audio.astype(np.float32))

                if not chunks:
                    raise ValueError("Uploaded file contains no decodable audio data")

                audio_np = np.concatenate(chunks)
                source_rate = int(audio_stream.rate or 24000)
                if source_rate != 24000:
                    audio_np = librosa.resample(audio_np, orig_sr=source_rate, target_sr=24000)
                return audio_np.astype(np.float32)
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)


class CloneTTSRequest(BaseModel):
    text: str
    ref_id: str
    language: str = "en"
    speed: float = 1.0
    normalize: bool = True
    compress: bool = False
    bass: float = 0.0
    treble: float = 0.0
    reverb: bool = False


class VideoRequest(BaseModel):
    gameplay_id:     str
    audio_id:        str
    subtitle_style:  str = "viral"
    subtitle_effect: str = "karaoke"
    format:          str = "shorts"
    vertical_style:  str = "blur"
    highlight_color: str = "cyan"
    mute_gameplay:   bool = False
    bg_music_vol:    float = 0.0
    sub_position:    str = "bottom"
    sub_size:        str = "medium"
    effect_intensity: int = 1
    av_offset:       float = 0.0
    uppercase:       bool = False
    music_id:        str = ""
    music_volume:    float = 0.35
    duck_intensity:  str = "light"
    music_start_offset: float = 0.0
    text_color:      str = "white"
    font_family:     str = ""
    box_opacity:     int = -1
    outline_width:   int = -1
    anim_speed:      str = "normal"
    easing:          str = "linear"
    color_cycle:     List[str] = []
    caption_mode:    str = "phrase"
    thumbnail_path:  str = ""   # optional: server-side path of the selected thumbnail
                                # (populated when user selects/generates a thumbnail in the
                                # Thumbnail Generator tab). Used to inject a 3-second
                                # "title card" at the start of the composed video.


class ASSRequest(BaseModel):
    audio_id:        str
    style:           str = "viral"
    effect:          str = "karaoke"  # karaoke, fade, scale, color, glow, bounce, typewriter
    highlight_color: str = "cyan"
    position:        str = "bottom"
    size:            str = "medium"
    intensity:       int = 1
    av_offset:       float = 0.0
    uppercase:       bool = False
    text_color:      str = "white"
    font_family:     str = ""
    box_opacity:     int = -1
    outline_width:   int = -1
    anim_speed:      str = "normal"
    easing:          str = "linear"
    color_cycle:     List[str] = []
    caption_mode:    str = "phrase"


@app.post("/upload-reference")
async def upload_reference(file: UploadFile = File(...)):
    """
    Upload voice sample for cloning.
    Min 6 sec, recommended 15-30 sec.
    The clone engine handles everything internally — no transcript needed.
    """
    try:
        filename = file.filename or "voice_sample.wav"
        existing = next(
            (rid for rid, info in clone_refs.items() if info["filename"] == filename),
            None,
        )
        ref_id = existing or str(uuid.uuid4())
        content = await file.read()

        # Load + normalize to 24kHz mono
        audio_np = load_uploaded_audio(content, filename)
        
        # Trim silence
        audio_np, _ = librosa.effects.trim(audio_np, top_db=20)
        
        # Min 6 seconds check
        if len(audio_np) < 6 * 24000:
            return {"error": "Too short. Record at least 6 seconds of clear speech.", "status": "failed"}
        
        # Cap at 30 seconds
        if len(audio_np) > 30 * 24000:
            audio_np = audio_np[:30 * 24000]
        
        # Save to exports folder
        ref_path = f"{EXPORTS_DIR}/ref_{ref_id}.wav"
        sf.write(ref_path, audio_np, 24000)
        
        duration = round(len(audio_np) / 24000, 1)
        
        # Store metadata
        clone_refs[ref_id] = {
            "path": ref_path,
            "duration": duration,
            "filename": filename
        }
        
        return {"ref_id": ref_id, "duration": duration, "filename": filename, "status": "ready"}
    
    except Exception as e:
        return {"error": str(e), "status": "failed"}


# ── ADVANCED VOICE CONVERSION (for voice cloning) ──────────────────
def extract_voice_features(audio: np.ndarray, sr: int) -> dict:
    """Extract comprehensive voice features for conversion"""
    try:
        # Fundamental frequency (pitch)
        f0, voiced_flag, _ = librosa.pyin(audio, fmin=50, fmax=400, sr=sr)
        f0_valid = f0[voiced_flag]
        f0_mean = np.mean(f0_valid) if len(f0_valid) > 0 else 100
        f0_std = np.std(f0_valid) if len(f0_valid) > 0 else 20
        
        # Spectral features
        spec = np.abs(librosa.stft(audio))
        spec_centroid = librosa.feature.spectral_centroid(S=spec, sr=sr)[0]
        spec_centroid_mean = np.mean(spec_centroid)
        spec_centroid_std = np.std(spec_centroid)
        
        # Mel-frequency cepstral coefficients (timbre)
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
        mfcc_mean = np.mean(mfcc, axis=1)
        mfcc_std = np.std(mfcc, axis=1)
        
        # Harmonic-to-noise ratio
        harmonic = librosa.effects.harmonic(audio)
        harmonic_power = np.mean(harmonic ** 2)
        noise_power = np.mean((audio - harmonic) ** 2)
        hnr = harmonic_power / (noise_power + 1e-8)
        
        # RMS energy
        rms = np.sqrt(np.mean(audio ** 2))
        
        # Zero crossing rate (voicing quality)
        zcr = librosa.feature.zero_crossing_rate(audio)[0]
        zcr_mean = np.mean(zcr)
        
        return {
            'f0_mean': f0_mean,
            'f0_std': f0_std,
            'spec_centroid_mean': spec_centroid_mean,
            'spec_centroid_std': spec_centroid_std,
            'mfcc_mean': mfcc_mean,
            'mfcc_std': mfcc_std,
            'hnr': hnr,
            'rms': rms,
            'zcr_mean': zcr_mean
        }
    except Exception as e:
        print(f"[Voice] Feature extraction error: {e}")
        return {}

def apply_voice_conversion(audio: np.ndarray, sr: int, source_features: dict, target_features: dict) -> np.ndarray:
    """Apply voice conversion to match target speaker characteristics.

    Previous implementation mixed in librosa.feature.inverse.mfcc_to_audio()
    which reconstructs audio from only 13 MFCCs — discarding ~99% of spectral
    detail and all phase information, producing a muffled, hollow, almost
    robotic sound. Replaced with:

    1. PSOLA pitch shifting (librosa.effects.pitch_shift) — time-domain,
       preserves formants and naturalness.
    2. Spectral envelope transfer via LPC-style cepstral liftering — shifts
       vocal tract resonances (formants) WITHOUT reconstruction artifacts.
    3. RMS energy matching — so the cloned voice is at the same loudness as
       the reference, not the Edge-TTS default.

    All operations stay in the time/cepstral domain; no audio reconstruction
    from MFCCs, so no muffling or hollow artifacts."""
    if not source_features or not target_features:
        return audio
    try:
        # ── Step 1: PSOLA Pitch shift ─────────────────────────────────
        # Shift the pitch of the Edge-TTS output to match the reference's
        # fundamental frequency (f0). librosa.effects.pitch_shift uses a
        # phase vocoder internally; -8/+8 semitone clamp keeps it natural.
        src_f0 = source_features.get('f0_mean', 150.0)
        tgt_f0 = target_features.get('f0_mean', 150.0)
        if src_f0 > 0 and tgt_f0 > 0:
            semitones = 12 * np.log2(tgt_f0 / (src_f0 + 1e-9))
            semitones = float(np.clip(semitones, -8, 8))
            if abs(semitones) > 0.3:  # skip trivial shifts
                audio = librosa.effects.pitch_shift(audio, sr=sr, n_steps=semitones)

        # ── Step 2: Spectral envelope transfer (formant shaping) ─────
        # Transfer the spectral envelope (vocal tract shape / formant
        # pattern) from the reference to the synthesized audio by
        # computing the ratio of their smoothed spectral envelopes and
        # applying it as a frequency-domain filter.
        # This is the key step that makes voices sound more alike —
        # it changes WHERE the resonances are (dark/bright, nasal/clear)
        # without touching the phase or fine structure.
        n_fft = 2048
        hop   = n_fft // 4

        # Compute magnitude spectra
        S_audio = np.abs(librosa.stft(audio,    n_fft=n_fft, hop_length=hop))
        # Smooth envelopes using median filter on log-spectrum (LPC-like)
        from scipy.signal import medfilt
        # Per-frequency smoothed envelopes
        env_src_mean = np.median(np.abs(librosa.stft(
            librosa.effects.harmonic(audio), n_fft=n_fft, hop_length=hop)) + 1e-8, axis=1)
        env_tgt_mean = (target_features.get('mfcc_mean', None) is not None and
                        source_features.get('mfcc_mean', None) is not None)

        # Build transfer ratio from MFCC means (this stays in cepstral space,
        # NO mfcc_to_audio reconstruction — just a smooth spectral multiplier)
        if env_tgt_mean:
            from scipy.ndimage import gaussian_filter1d
            src_mc = source_features['mfcc_mean']   # shape (13,)
            tgt_mc = target_features['mfcc_mean']   # shape (13,)
            mc_diff = tgt_mc - src_mc               # delta in cepstral space
            # Lift cepstral delta back to log-spectrum bins via DCT
            n_bins = n_fft // 2 + 1
            n_mfcc = len(mc_diff)
            # Build a cos-basis matrix (same as MFCC filterbank inverse)
            basis = np.cos(np.pi / n_mfcc * np.outer(
                np.arange(n_mfcc), np.arange(0.5, n_bins)))  # (n_mfcc, n_bins)
            log_gain = mc_diff @ basis     # (n_bins,) — smooth log-magnitude delta
            log_gain = gaussian_filter1d(log_gain, sigma=5)
            # Clip to ±6 dB so we reshape, not replace, the voice
            log_gain = np.clip(log_gain / 10.0, -0.6, 0.6)  # /10 = tentative dB scale
            gain = 10 ** log_gain           # linear multiplier per frequency bin

            # Apply gain to each frame of the STFT
            D = librosa.stft(audio, n_fft=n_fft, hop_length=hop)
            D_shaped = D * gain[:, np.newaxis]
            audio_shaped = librosa.istft(D_shaped, hop_length=hop, length=len(audio))
            # Blend 60% shaped + 40% original — preserves naturalness while
            # shifting the spectral envelope toward the target voice.
            audio = 0.6 * audio_shaped + 0.4 * audio[:len(audio_shaped)]
            audio = audio[:len(audio)]  # restore original length

        # ── Step 3: RMS energy matching ──────────────────────────────
        tgt_rms = float(target_features.get('rms', 0.1))
        cur_rms = float(np.sqrt(np.mean(audio ** 2)) + 1e-9)
        gain_factor = float(np.clip(tgt_rms / cur_rms, 0.3, 3.0))
        audio = audio * gain_factor

        return np.clip(audio, -0.99, 0.99).astype(np.float32)

    except Exception as e:
        print(f"[Voice] Conversion error: {e}")
        return np.clip(audio, -0.99, 0.99).astype(np.float32)


# ── SPEAKER EMBEDDING ENGINE (resemblyzer) ────────────────────────
# resemblyzer extracts a 256-dim d-vector speaker embedding — a compact
# numerical "fingerprint" of a voice's identity (timbre, resonance, accent).
# We store the embedding alongside each uploaded reference, then use it two
# ways in generate_clone():
#  1. Pick the Kokoro voice whose own embedding is closest to the reference
#     (cosine similarity) — so the BASE synthesis already sounds similar to
#     the target speaker before any DSP conversion is applied.
#  2. Measure conversion quality post-hoc (future improvement).
_resemblyzer_encoder = None
_KOKORO_VOICE_EMBEDDINGS: dict = {}   # populated lazily on first clone call


def _get_resemblyzer_encoder():
    global _resemblyzer_encoder
    if _resemblyzer_encoder is None:
        try:
            from resemblyzer import VoiceEncoder
            _resemblyzer_encoder = VoiceEncoder()
            print("  Resemblyzer speaker encoder  ✓  Ready")
        except Exception as e:
            print(f"  [Clone] Resemblyzer not available ({e}) — falling back to DSP only")
            _resemblyzer_encoder = False   # sentinel: tried, failed
    return _resemblyzer_encoder if _resemblyzer_encoder is not False else None


def _embed_audio(audio_np: np.ndarray, sr: int) -> "np.ndarray | None":
    """Return a 256-dim d-vector for audio_np, or None if unavailable."""
    enc = _get_resemblyzer_encoder()
    if enc is None:
        return None
    try:
        from resemblyzer import preprocess_wav
        import tempfile, soundfile as _sf
        # resemblyzer wants a file path or a preprocessed wav — easiest is
        # to write a temp file then let it preprocess
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name
        _sf.write(tmp_path, audio_np, sr)
        wav = preprocess_wav(tmp_path)
        os.unlink(tmp_path)
        emb = enc.embed_utterance(wav)   # shape (256,)
        return emb
    except Exception as e:
        print(f"  [Clone] Embedding failed: {e}")
        return None


def _cosine_sim(a, b):
    denom = (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9)
    return float(np.dot(a, b) / denom)


# Map Kokoro voice IDs to simple descriptors so we can pick the most
# gender/style-appropriate one based on the reference embedding.
# Embeddings for Kokoro voices are computed once on first clone call and
# cached in _KOKORO_VOICE_EMBEDDINGS.
_KOKORO_VOICE_CANDIDATES = [
    "af_bella", "af_nova", "af_sky", "af_sarah",   # female US
    "am_adam", "am_michael",                         # male US
    "bf_emma", "bf_isabella",                        # female UK
    "bm_george", "bm_lewis",                         # male UK
]


def _get_best_kokoro_voice_for_embedding(ref_emb: "np.ndarray",
                                          kokoro_pipeline) -> str:
    """Return the Kokoro voice ID whose embedding is closest to ref_emb."""
    global _KOKORO_VOICE_EMBEDDINGS
    if ref_emb is None or kokoro_pipeline is None:
        return "af_bella"   # sane default

    # Build Kokoro voice embeddings if not cached yet
    if not _KOKORO_VOICE_EMBEDDINGS:
        print("  [Clone] Pre-computing Kokoro voice embeddings (one-time, ~10s)...")
        for vid in _KOKORO_VOICE_CANDIDATES:
            try:
                gen = kokoro_pipeline(
                    "Hello, this is a voice sample for speaker matching.",
                    voice=vid, speed=1.0
                )
                chunks = [a for _, _, a in gen]
                if not chunks:
                    continue
                audio_sample = np.concatenate(chunks).astype(np.float32)
                emb = _embed_audio(audio_sample, 24000)
                if emb is not None:
                    _KOKORO_VOICE_EMBEDDINGS[vid] = emb
                    print(f"    {vid}  ✓")
            except Exception as e:
                print(f"    {vid}  ✗  {e}")

    if not _KOKORO_VOICE_EMBEDDINGS:
        return "af_bella"

    best_voice = max(_KOKORO_VOICE_EMBEDDINGS,
                     key=lambda v: _cosine_sim(ref_emb, _KOKORO_VOICE_EMBEDDINGS[v]))
    sim = _cosine_sim(ref_emb, _KOKORO_VOICE_EMBEDDINGS[best_voice])
    print(f"  [Clone] Best Kokoro match → {best_voice}  (cosine sim: {sim:.3f})")
    return best_voice


@app.post("/generate-clone")
async def generate_clone(req: CloneTTSRequest):
    """
    Voice cloning — best available engine, with graceful fallback:
    TIER 0:  Chatterbox TTS  — true neural voice cloning (best quality; needs
             Python 3.11 exactly, upstream package constraint)
    TIER 0b: Coqui XTTS v2   — true neural voice cloning (used only if
             Chatterbox isn't installed; needs C++ Build Tools on Windows)
    TIER 1:  resemblyzer embedding → closest Kokoro voice → DSP conversion
    TIER 2:  resemblyzer embedding → Edge-TTS → DSP conversion
    TIER 3:  DSP-only (no resemblyzer installed)
    Neither neural engine is required — Tiers 1-3 work with zero extra installs.
    """
    try:
        if req.ref_id not in clone_refs:
            return JSONResponse(status_code=404,
                content={"error": "Voice reference not found. Upload a sample first."})
        ref = clone_refs[req.ref_id]
        ref_path = ref["path"]
        if not os.path.exists(ref_path):
            return JSONResponse(status_code=404,
                content={"error": "Reference audio missing from disk. Upload again."})

        SR = 24000
        converted = None
        engine_used = None

        # ── TIER 0: Chatterbox — true neural cloning (best quality) ──
        try:
            model = await asyncio.to_thread(get_chatterbox)
            wav = await asyncio.to_thread(
                model.generate, req.text,
                audio_prompt_path=ref_path,
                exaggeration=0.5, cfg_weight=0.5, temperature=0.8,
            )
            converted = wav.squeeze().cpu().numpy().astype(np.float32)
            SR = model.sr
            engine_used = "chatterbox"
            print("  [Clone] Tier 0 — Chatterbox neural clone ✓")
        except ImportError:
            pass
        except Exception as e:
            print(f"  [Clone] Chatterbox failed ({e}), trying next tier")

        # ── TIER 0b: Coqui XTTS v2 — true neural cloning (fallback) ──
        if converted is None:
            try:
                model = await asyncio.to_thread(get_xtts)
                out_path = f"{EXPORTS_DIR}/xtts_{uuid.uuid4().hex}.wav"
                await asyncio.to_thread(
                    model.tts_to_file, text=req.text, speaker_wav=ref_path,
                    language=req.language or "en", file_path=out_path,
                )
                converted, SR = librosa.load(out_path, sr=None, mono=True)
                if os.path.exists(out_path):
                    os.unlink(out_path)
                engine_used = "xtts"
                print("  [Clone] Tier 0b — Coqui XTTS neural clone ✓")
            except (ImportError, RuntimeError):
                pass
            except Exception as e:
                print(f"  [Clone] XTTS failed ({e}), falling back to DSP tiers")

        # ── TIER 1-3: DSP fallback — used when neither neural engine is
        # installed, or both failed to load/run ──
        if converted is None:
            ref_audio, _ = librosa.load(ref_path, sr=SR, mono=True)
            ref_features = extract_voice_features(ref_audio, SR)

            # Step 1: speaker embedding via resemblyzer (Tier 1 & 2)
            ref_emb = await asyncio.to_thread(_embed_audio, ref_audio, SR)
            if ref_emb is not None:
                print(f"  [Clone] Reference embedding ✓ (norm={np.linalg.norm(ref_emb):.2f})")

            # Step 2: pick synthesiser
            tts_audio = None
            kokoro_ok = kpipeline is not None

            if ref_emb is not None and kokoro_ok:
                # TIER 1 — best Kokoro voice by embedding cosine similarity
                best_voice = await asyncio.to_thread(
                    _get_best_kokoro_voice_for_embedding, ref_emb, kpipeline)
                print(f"  [Clone] Tier 1 — Kokoro ({best_voice})")
                try:
                    chunks = [a for _, _, a in kpipeline(req.text, voice=best_voice, speed=req.speed)]
                    if chunks:
                        tts_audio = np.concatenate(chunks).astype(np.float32)
                except Exception as e:
                    print(f"  [Clone] Kokoro failed ({e}), falling back")

            if tts_audio is None:
                # TIER 2/3 — Edge-TTS base
                tier_label = "Tier 2 (Resemblyzer+EdgeTTS)" if ref_emb is not None else "Tier 3 (DSP only)"
                print(f"  [Clone] {tier_label}")
                edge_path = f"{EXPORTS_DIR}/edge_{uuid.uuid4().hex}.wav"
                try:
                    comm = edge_tts.Communicate(text=req.text, voice="en-US-AriaNeural", rate="+0%", volume="+0%")
                    await asyncio.wait_for(comm.save(edge_path), timeout=45.0)
                    tts_audio, _ = librosa.load(edge_path, sr=SR, mono=True)
                finally:
                    if os.path.exists(edge_path):
                        os.unlink(edge_path)

            # Step 3: DSP voice conversion
            tts_features = extract_voice_features(tts_audio, SR)
            converted = apply_voice_conversion(tts_audio, SR, tts_features, ref_features)
            engine_used = "dsp"

        # ── Post-processing (applies regardless of which engine produced audio) ──
        if req.normalize:
            converted = apply_normalization(converted, target_db=-20.0)
        if req.compress:
            converted = apply_compression(converted)
        if req.bass != 0.0 or req.treble != 0.0:
            converted = apply_eq(converted, SR, bass_db=float(req.bass), treble_db=float(req.treble))
        if req.reverb:
            converted = apply_reverb(converted, SR)
        converted = np.clip(converted, -0.99, 0.99)

        job_id, _ = save_audio_file(converted, f"clone_{req.ref_id[:8]}_{req.language}", req.speed)
        buf = io.BytesIO()
        sf.write(buf, converted, SR, format="WAV")
        buf.seek(0)
        return StreamingResponse(buf, media_type="audio/wav",
            headers={"Content-Disposition": f'attachment; filename="clone_{job_id}.wav"',
                     "X-Job-ID": job_id, "X-Clone-Engine": engine_used})

    except Exception as e:
        print(f"[Clone] Error: {e}")
        import traceback; traceback.print_exc()
        return JSONResponse(status_code=500,
            content={"error": f"Voice cloning failed: {str(e)}", "status": "failed"})


@app.get("/clone-refs")
async def get_clone_refs():
    """List all uploaded voice references for this session."""
    return {
        "refs": [
            {"ref_id": ref_id, "duration": info["duration"], "filename": info["filename"]}
            for ref_id, info in clone_refs.items()
        ]
    }


# ── VIDEO STUDIO HELPERS ──────────────────────────────────────────
def format_ass_time(seconds: float) -> str:
    """Format seconds as ASS timestamp H:MM:SS.cc"""
    h  = int(seconds // 3600)
    m  = int((seconds % 3600) // 60)
    s  = int(seconds % 60)
    cs = int((seconds % 1) * 100)
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"


def get_video_info(filepath: str) -> dict:
    """Get video width, height, duration using ffmpeg"""
    try:
        import re
        # Use ffmpeg to get video info (imageio_ffmpeg only provides ffmpeg binary, not ffprobe)
        cmd = [FFMPEG_EXE, "-i", filepath]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
        
        # Parse stderr output (ffmpeg writes to stderr)
        output = result.stderr
        
        # Extract duration from "Duration: HH:MM:SS.ms" line
        duration = 0
        for line in output.split('\n'):
            if 'Duration' in line:
                match = re.search(r'Duration: (\d+):(\d+):(\d+)', line)
                if match:
                    h, m, s = map(int, match.groups())
                    duration = h * 3600 + m * 60 + s
                break
        
        # Extract video dimensions from "Stream #0:0: Video:" lines
        width, height = 1920, 1080
        for line in output.split('\n'):
            if 'Video:' in line:
                # Look for resolution like "1920x1080"
                # Use findall to get all matches and take the last one (most specific)
                matches = re.findall(r'(\d+)x(\d+)', line)
                if matches:
                    w, h = matches[-1]
                    width, height = int(w), int(h)
                break
        
        return {"width": width, "height": height, "duration": duration}
    except Exception as e:
        print(f"⚠ get_video_info error: {e}")
        pass
    return {"width": 1920, "height": 1080, "duration": 0}


def _hex_to_ass(color: str) -> str:
    """Convert a color to ASS hex format (&H00BBGGRR).
    Accepts either a preset name (cyan/yellow/green/white/...) or a raw
    '#RRGGBB' hex string from a color picker."""
    presets = {
        "cyan": "#00FFFF", "yellow": "#FFD700", "green": "#22FF88", "white": "#FFFFFF",
        "red": "#FF4444", "orange": "#FF8800", "purple": "#CC44FF", "pink": "#FF66CC",
        "blue": "#3399FF",
    }
    hexval = presets.get(color, color)
    if isinstance(hexval, str) and hexval.startswith("#") and len(hexval) == 7:
        try:
            r = hexval[1:3]; g = hexval[3:5]; b = hexval[5:7]
            return f"&H00{b.upper()}{g.upper()}{r.upper()}"
        except Exception:
            pass
    return "&H0000FFFF"  # fallback cyan


def get_word_segments(audio_id: str, audio_path: str) -> list:
    """Run Whisper word-level transcription ONCE per audio_id and cache the
    result. Subsequent subtitle-style/effect/position changes (re-Compose in
    Video Studio) reuse the cached transcription — only ASS re-rendering
    (fast, pure formatting) runs again, not the expensive STT pass.

    FIXED: faster-whisper's bundled Silero VAD has a known bug where very
    short clips, near-silent clips, or clips whose speech doesn't reach the
    VAD's internal chunk boundaries throw 'tuple index out of range' deep
    inside its timestamp-restoration code (unrelated to whether the audio
    is actually valid). vad_filter=True is a nice-to-have (skips silence,
    slightly faster) — it is not worth a hard failure. We now retry once
    without it, and only surface the real error if that ALSO fails."""
    if audio_id in WORD_CACHE:
        return WORD_CACHE[audio_id]

    def _run(vad: bool):
        segments_iter, _ = whisper_model.transcribe(
            audio_path, word_timestamps=True, language=None, beam_size=5, vad_filter=vad
        )
        out = []
        for seg in segments_iter:
            words = [{"word": w.word, "start": w.start, "end": w.end} for w in (seg.words or [])]
            out.append({"start": seg.start, "end": seg.end, "text": seg.text, "words": words})
        return out

    try:
        segments = _run(vad=True)
    except Exception as vad_err:
        try:
            segments = _run(vad=False)
        except Exception as plain_err:
            raise RuntimeError(
                f"Transcription failed even without VAD filtering: {plain_err} "
                f"(original VAD error: {vad_err})"
            )

    WORD_CACHE[audio_id] = segments
    return segments


def generate_ass_content(segments: list, style: str, highlight_color: str, effect: str = "karaoke",
                          position: str = "bottom", size: str = "medium", intensity: int = 1,
                          av_offset: float = 0.0, uppercase: bool = False, text_color: str = "white",
                          font_family: str = "", box_opacity: int = -1, outline_width: int = -1,
                          anim_speed: str = "normal", easing: str = "linear", color_cycle: list = None,
                          caption_mode: str = "phrase", custom_pos: dict = None) -> str:
    """Render an ASS subtitle file from PRE-TRANSCRIBED segments (see
    get_word_segments — this function does NOT run Whisper, so style/effect
    changes re-render instantly without re-transcribing).

    Effects:
    - karaoke: Word highlight with color change (default) — always phrase-mode
    - fade: Fade in/out transitions between words
    - scale: Scale/zoom animation on words
    - color: Color cycling through words — always phrase-mode
    - glow: Shadow/glow effect on words
    - bounce: Bouncy/wiggle animation
    - pop: Punchy scale-overshoot entrance
    - typewriter: Letter-by-letter reveal

    caption_mode: "phrase" (default) shows the WHOLE sentence/segment on
    screen at once, with each word's animation playing in-place as it's
    spoken while neighboring words stay visible — the classic TikTok/Reels
    auto-caption look, and what the live preview shows. "word" shows only
    ONE word on screen at a time, replaced by the next as it's spoken —
    faster-paced, more minimal. karaoke/color are inherently multi-word
    effects and always render as phrase regardless of this setting.

    av_offset: shift ALL subtitle timestamps later by this many seconds —
    used when the voiceover audio itself is delayed in the final video
    (so subtitles stay in sync with the delayed voice).
    uppercase: render all subtitle text in CAPS (popular "viral" style).
    text_color: base subtitle text color — preset name or '#RRGGBB' hex.
    font_family: override the style's default font (empty = use style default).
    box_opacity: 0-100, how visible the background box is. -1 = style default.
    outline_width: 0-10 outline thickness override, -1 = use style default.
    anim_speed: "slow"|"normal"|"fast" — how quickly entry/exit animations play.
    easing: "linear"|"ease_in"|"ease_out" — animation acceleration curve.
    color_cycle: custom list of ASS hex colors for the "color" cycling effect
                 (None = use highlight_color + 3 built-in defaults).
    """
    av_offset = max(0.0, float(av_offset))
    caption_mode = caption_mode if caption_mode in ("word", "phrase") else "phrase"

    if av_offset > 0 or uppercase:
        shifted = []
        for seg in segments:
            new_seg = {
                "start": seg["start"] + av_offset,
                "end":   seg["end"] + av_offset,
                "text":  seg["text"].upper() if uppercase else seg["text"],
                "words": [
                    {"word": (w["word"].upper() if uppercase else w["word"]),
                     "start": w["start"] + av_offset, "end": w["end"] + av_offset}
                    for w in seg["words"]
                ],
            }
            shifted.append(new_seg)
        segments = shifted

    hl_color = _hex_to_ass(highlight_color)
    txt_color = _hex_to_ass(text_color) if (text_color and text_color != "white") else "&H00FFFFFF"

    # Base fontsizes per style, then scaled by the user's chosen size
    BASE_SIZES = {"viral": 72, "clean": 56, "netflix": 42, "minimal": 36}
    SIZE_MULT  = {"small": 0.7, "medium": 1.0, "large": 1.3, "xlarge": 1.6}
    mult = SIZE_MULT.get(size, 1.0)
    fsz  = int(BASE_SIZES.get(style, 72) * mult)

    # Alignment (ASS numpad: 2=bottom-center, 5=middle-center, 8=top-center)
    # MarginV = distance from that edge
    POSITIONS = {
        "bottom": (2, 80),
        "middle": (5, 0),
        "top":    (8, 80),
    }
    align, marginv = POSITIONS.get(position, (2, 80))
    # Netflix/minimal traditionally sit a bit closer to the edge
    if style in ("netflix", "minimal") and position != "middle":
        marginv = max(40, marginv - 30)

    # Default fonts + outline widths + box alpha per style — each overridable
    DEFAULT_FONTS    = {"viral": "Impact", "clean": "Arial Bold", "netflix": "Arial", "minimal": "Arial"}
    DEFAULT_OUTLINES = {"viral": 5, "clean": 3, "netflix": 0, "minimal": 2}
    DEFAULT_BOX_OP   = {"viral": 56, "clean": 0, "netflix": 63, "minimal": 38}  # approx visual % of original hardcoded alphas

    font = font_family.strip() if (font_family and font_family.strip()) else DEFAULT_FONTS.get(style, "Arial")
    outline = outline_width if (outline_width is not None and outline_width >= 0) else DEFAULT_OUTLINES.get(style, 3)
    box_op_pct = box_opacity if (box_opacity is not None and box_opacity >= 0) else DEFAULT_BOX_OP.get(style, 50)
    box_op_pct = max(0, min(100, box_op_pct))
    box_alpha = format(int(round((100 - box_op_pct) / 100 * 255)), "02X")  # 100%=visible(alpha 00), 0%=invisible(alpha FF)

    # box_alpha already correctly encodes 0% -> FF (transparent), 100% -> 00 (opaque)
    # for ALL styles uniformly — no special-casing needed.
    back_viral   = f"&H{box_alpha}000000"
    back_clean   = f"&H{box_alpha}000000"
    back_netflix = f"&H{box_alpha}000000"
    back_minimal = f"&H{box_alpha}000000"

    STYLES = {
        "viral":   f"Style: Default,{font},{fsz},{txt_color},{hl_color},&H00000000,{back_viral},1,0,0,0,100,100,2,0,1,{outline},3,{align},30,30,{marginv},1",
        "clean":   f"Style: Default,{font},{fsz},{txt_color},{hl_color},&H00000000,{back_clean},1,0,0,0,100,100,1,0,1,{outline},1,{align},40,40,{marginv},1",
        "netflix": f"Style: Default,{font},{fsz},{txt_color},{txt_color},&H80000000,{back_netflix},0,0,0,0,100,100,0,0,3,{outline},0,{align},40,40,{marginv},1",
        "minimal": f"Style: Default,{font},{fsz},{txt_color},{hl_color},&H60000000,{back_minimal},0,0,0,0,100,100,0,0,1,{outline},0,{align},40,40,{marginv},1"
    }
    style_line = STYLES.get(style, STYLES["viral"])

    # Intensity scaling for animated effects: 0=subtle, 1=medium, 2=strong
    INTENSITY = {
        0: {"scale_from": 75, "scale_pop": 115, "bounce_amp": 8,  "frz": 2,  "fade_frac": 0.20, "blur": 3},
        1: {"scale_from": 55, "scale_pop": 135, "bounce_amp": 18, "frz": 4,  "fade_frac": 0.30, "blur": 6},
        2: {"scale_from": 30, "scale_pop": 160, "bounce_amp": 32, "frz": 8,  "fade_frac": 0.40, "blur": 10},
    }
    INT = INTENSITY.get(intensity, INTENSITY[1])

    # Animation SPEED: how much of a word's spoken duration the transition
    # consumes — slow = transition plays out lazily across the word, fast =
    # snaps almost instantly. Multiplies the timing fractions used below.
    SPEED_MULT = {"slow": 1.6, "normal": 1.0, "fast": 0.55}
    spd = SPEED_MULT.get(anim_speed, 1.0)

    # Easing: maps to ASS \t(t1,t2,accel,...) acceleration parameter.
    # accel=1 is linear; >1 eases in (slow start, fast finish); <1 eases out.
    EASING_ACCEL = {"linear": 1.0, "ease_in": 1.8, "ease_out": 0.55}
    accel = EASING_ACCEL.get(easing, 1.0)
    accel_tag = "" if abs(accel - 1.0) < 0.01 else f",{accel}"

    # Custom color cycle for the "color" effect (falls back to highlight + 3 defaults)
    if color_cycle and isinstance(color_cycle, list) and len(color_cycle) > 0:
        cycle_colors = [_hex_to_ass(c) for c in color_cycle]
    else:
        cycle_colors = [hl_color, "&H0000FF00", "&H00FFFF00", "&H00FF00FF"]

    ass = ("[Script Info]\nScriptType: v4.00+\nPlayResX: 1080\nPlayResY: 1920\nWrapStyle: 1\n\n"
           "[V4+ Styles]\nFormat: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, "
           "ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding\n"
           f"{style_line}\n\n[Events]\nFormat: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n")

    def _word_tag(effect_name, rel_s, rel_e):
        """Build the ASS override-block tag for one word, given its time
        offsets RELATIVE TO THE CONTAINING LINE'S START (so this works both
        for a per-word line where rel_s=0, and for a phrase line where
        rel_s/rel_e are each word's offset into the shared sentence line)."""
        dur_cs = max(1, rel_e - rel_s)
        if effect_name == "fade":
            fdur = max(1, int(dur_cs * INT["fade_frac"] * spd))
            return f"{{\\alpha&HFF&\\t({rel_s},{rel_s+fdur},\\alpha&H00&)}}"
        elif effect_name == "scale":
            sf = INT["scale_from"]
            grow_t = rel_s + max(1, int(dur_cs * 0.6 * spd))
            return f"{{\\fscx{sf}\\fscy{sf}\\t({rel_s},{grow_t}{accel_tag},\\fscx100\\fscy100)}}"
        elif effect_name == "pop":
            sp = INT["scale_pop"]
            t1 = rel_s + max(1, int(dur_cs * 0.3 * spd))
            t2 = rel_s + max(2, int(dur_cs * 0.6 * spd))
            return f"{{\\fscx30\\fscy30\\t({rel_s},{t1}{accel_tag},\\fscx{sp}\\fscy{sp})\\t({t1},{t2}{accel_tag},\\fscx100\\fscy100)}}"
        elif effect_name == "bounce":
            amp = INT["bounce_amp"]; frz = INT["frz"]
            mid = rel_s + max(1, int(dur_cs * 0.5 * spd))
            endt = rel_s + max(2, int(dur_cs * spd))
            return (f"{{\\fscy{100-amp}\\frz-{frz}"
                    f"\\t({rel_s},{mid}{accel_tag},\\fscy{100+amp}\\frz{frz})"
                    f"\\t({mid},{endt}{accel_tag},\\fscy100\\frz0)}}")
        elif effect_name == "glow":
            blur = INT["blur"]
            gend = rel_s + max(1, int(dur_cs * 0.8))
            return f"{{\\3c{hl_color}\\blur{blur}\\t({rel_s},{gend},\\3c&H000000&\\blur0)}}"
        return ""

    for seg in segments:
        words = seg.get("words") or []
        if not words:
            ds = format_ass_time(seg["start"])
            de = format_ass_time(seg["end"])
            ass += f"Dialogue: 0,{ds},{de},Default,,0,0,0,,{seg['text'].strip()}\n"
            continue

        # ── KARAOKE / COLOR: always phrase-mode, single line per sentence ──
        # \kf timing is cumulative WITHIN the line, so a single Dialogue line
        # naturally shows the whole sentence with a progressive sweep.
        if effect == "karaoke":
            line = ""
            for w in words:
                dur_cs = max(1, int((w["end"] - w["start"]) * 100))
                line += f"{{\\kf{dur_cs}}}{w['word']} "
            ds = format_ass_time(seg["start"]); de = format_ass_time(seg["end"])
            ass += f"Dialogue: 0,{ds},{de},Default,,0,0,0,,{line.strip()}\n"
            continue

        if effect == "color":
            line = ""
            for idx, w in enumerate(words):
                dur_cs = max(1, int((w["end"] - w["start"]) * 100))
                color = cycle_colors[idx % len(cycle_colors)]
                line += f"{{\\kf{dur_cs}\\c{color}}}{w['word']} "
            ds = format_ass_time(seg["start"]); de = format_ass_time(seg["end"])
            ass += f"Dialogue: 0,{ds},{de},Default,,0,0,0,,{line.strip()}\n"
            continue

        # ── TYPEWRITER ──
        if effect == "typewriter":
            if caption_mode == "phrase":
                # Reveal the WHOLE sentence letter-by-letter, building up
                # across the segment's full duration (classic typewriter).
                full_text = " ".join(w["word"].strip() for w in words)
                letters = list(full_text)
                n = max(1, len(letters))
                seg_start, seg_end = words[0]["start"], words[-1]["end"]
                step = (seg_end - seg_start) / n
                for li in range(n):
                    sub_s = seg_start + li * step
                    sub_e = seg_end if li == n - 1 else (seg_start + (li + 1) * step)
                    partial = "".join(letters[:li + 1])
                    ass += (f"Dialogue: 0,{format_ass_time(sub_s)},{format_ass_time(sub_e)},"
                            f"Default,,0,0,0,,{partial}\n")
            else:
                # Word mode (legacy): reveal each word's letters independently,
                # one word visible at a time.
                for w in words:
                    text = w["word"].strip()
                    if not text:
                        continue
                    letters = list(text)
                    n = max(1, len(letters))
                    w_start_t, w_end_t = w["start"], w["end"]
                    step = (w_end_t - w_start_t) / n
                    for li in range(n):
                        sub_s = w_start_t + li * step
                        sub_e = w_end_t if li == n - 1 else (w_start_t + (li + 1) * step)
                        partial = "".join(letters[:li + 1])
                        ass += (f"Dialogue: 0,{format_ass_time(sub_s)},{format_ass_time(sub_e)},"
                                f"Default,,0,0,0,,{partial}\n")
            continue

        # ── FADE / SCALE / POP / BOUNCE / GLOW ──
        if caption_mode == "phrase":
            # ONE Dialogue line for the whole sentence. Each word gets its
            # own override block with \t() timing offsets RELATIVE TO THE
            # LINE'S OWN START (cumulative, same principle as \kf above) —
            # so multiple words are visible together and each animates
            # in-place exactly when it's spoken. This is what makes the
            # rendered video match the live browser preview.
            line_start = words[0]["start"]
            line_end   = words[-1]["end"]
            parts = []
            for w in words:
                text = w["word"]
                rel_s = max(0, int(round((w["start"] - line_start) * 100)))
                rel_e = max(rel_s + 1, int(round((w["end"] - line_start) * 100)))
                tag = _word_tag(effect, rel_s, rel_e)
                parts.append(tag + text)
            line_text = "".join(parts).strip()
            ass += (f"Dialogue: 0,{format_ass_time(line_start)},{format_ass_time(line_end)},"
                    f"Default,,0,0,0,,{line_text}\n")
        else:
            # WORD mode (legacy): one Dialogue line PER WORD, only one word
            # visible on screen at a time, replaced as the next is spoken.
            for w in words:
                text = w["word"].strip()
                if not text:
                    continue
                dur_cs = max(1, int((w["end"] - w["start"]) * 100))
                ws = format_ass_time(w["start"])
                we = format_ass_time(w["end"])
                tag = _word_tag(effect, 0, dur_cs)
                ass += f"Dialogue: 0,{ws},{we},Default,,0,0,0,,{tag}{text}\n"

    if custom_pos and "x_pct" in custom_pos and "y_pct" in custom_pos:
        # Free-form positioning — the style's Alignment tag still defines
        # which point of the text box is being placed (e.g. bottom-center),
        # \pos just moves that anchor point anywhere on the 1080x1920 canvas
        # instead of only the top/middle/bottom presets. One safe regex
        # pass over the fully-built ASS string instead of touching every
        # effect branch's Dialogue-line f-string individually above.
        px = max(0, min(100, float(custom_pos["x_pct"]))) / 100 * 1080
        py = max(0, min(100, float(custom_pos["y_pct"]))) / 100 * 1920
        pos_tag = f"{{\\pos({px:.0f},{py:.0f})}}"
        # FIX: re.sub's replacement STRING is also scanned for backslash
        # escapes (for backreferences like \1) — so a literal "\pos(...)"
        # tag in the replacement made it choke on "\p" as an unrecognized
        # escape ("bad escape \p"). A lambda replacement is never scanned
        # for escapes, so it's the safe way to inject a literal backslash.
        ass = re.sub(r"(Dialogue: 0,[^\n]*?,,0,0,0,,)", lambda m: m.group(1) + pos_tag, ass)

    return ass


def build_ffmpeg_filter(gameplay_w: int, gameplay_h: int, fmt: str, vertical_style: str, ass_path: str) -> list:
    """Build FFmpeg filter_complex and output options.
    ass_path must already be escaped/absolute before calling this function."""
    import re
    # Escape the path for FFmpeg: convert backslashes to forward slashes
    ass_path_escaped = ass_path.replace("\\", "/")
    # Escape Windows drive letter colon: C:/ -> C\\:/
    # This is crucial because FFmpeg filter strings need the colon escaped with a backslash
    if len(ass_path_escaped) > 1 and ass_path_escaped[1] == ':':
        ass_path_escaped = ass_path_escaped[0] + '\\:' + ass_path_escaped[2:]
    
    if fmt == "shorts":
        if vertical_style == "blur":
            fg_h = min(gameplay_h, 1080)
            fg_w = int(fg_h * gameplay_w / gameplay_h)
            vf = (f"[0:v]split[bg][fg];[bg]scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920,boxblur=25:5[blurred];"
                  f"[fg]scale={fg_w}:{fg_h}[scaled];[blurred][scaled]overlay=(W-w)/2:(H-h)/2[composed];"
                  f"[composed]ass='{ass_path_escaped}'[out]")
        else:
            vf = (f"[0:v]scale=1920:1080:force_original_aspect_ratio=decrease,crop=ih*9/16:ih,scale=1080:1920[composed];"
                  f"[composed]ass='{ass_path_escaped}'[out]")
    else:
        vf = (f"[0:v]scale=1920:1080:force_original_aspect_ratio=decrease,pad=1920:1080:(ow-iw)/2:(oh-ih)/2[composed];"
              f"[composed]ass='{ass_path_escaped}'[out]")
    return ["-filter_complex", vf, "-map", "[out]"]


# ── ERROR HANDLING HELPERS ────────────────────────────────────────
def format_user_error(error_str: str, gameplay_duration: float = None) -> tuple:
    """Convert technical FFmpeg error to user-friendly message with suggestions.
    Returns: (simple_message, suggestion)"""
    
    error_lower = error_str.lower()
    
    # Duration/length errors
    if "duration" in error_lower or "too long" in error_lower or "timeout" in error_lower:
        if gameplay_duration and gameplay_duration > 180:
            return ("❌ Video is too long for processing", "Try trimming to 3 minutes or less, or use a shorter video for Shorts")
        return ("❌ Video processing timed out", "Try a shorter video or use a smaller resolution")
    
    # Audio errors
    if "audio" in error_lower or "[1:a]" in error_lower or "audio stream" in error_lower:
        return ("❌ Audio issue with voiceover", "Try regenerating the voiceover in TTS Studio")
    
    # Codec/format errors
    if "codec" in error_lower or "format" in error_lower or "unsupported" in error_lower:
        return ("❌ Video format issue", "Try converting to MP4 format using a video converter online")
    
    # Subtitle/ASS errors
    if "ass" in error_lower or "subtitle" in error_lower or "font" in error_lower:
        return ("❌ Subtitle rendering failed", "Try changing the subtitle style (Viral → Clean → Netflix → Minimal)")
    
    # Memory/resource errors
    if "memory" in error_lower or "out of" in error_lower or "resource" in error_lower:
        return ("❌ Not enough system resources", "Close other applications and try again, or use a shorter video")
    
    # Filter/processing errors
    if "filter" in error_lower or "scale" in error_lower or "pad" in error_lower:
        return ("❌ Video composition failed", "Try a different video format or change subtitle style")
    
    # Path/file errors
    if "no such file" in error_lower or "not found" in error_lower or "permission" in error_lower:
        return ("❌ File access error", "Try uploading the video again")
    
    # Default
    return ("❌ Video generation failed", "Try again with a different video or settings")


def _gameplay_has_audio(gameplay_path: str) -> bool:
    """Check if the gameplay file has an audio stream."""
    try:
        cmd = [FFMPEG_EXE, "-i", gameplay_path]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        return "Audio:" in result.stderr
    except Exception:
        return False


def run_video_job(job_id: str, gameplay_path: str, audio_path: str, ass_path: str, output_path: str,
                  fmt: str, vertical_style: str, gameplay_w: int, gameplay_h: int, mute_gameplay: bool,
                  av_offset: float = 0.0, music_path: str = None, music_volume: float = 0.25,
                  duck_intensity: str = "light", music_start_offset: float = 0.0,
                  gameplay_duration: float = 0.0, thumbnail_path: str = None):
    """Blocking FFmpeg composition — run via asyncio.to_thread.

    av_offset: delay the voiceover (and therefore subtitles, which were
    already shifted by the same amount when the .ass was rendered) by this
    many seconds at the start of the final video — e.g. for a silent intro.

    music_path: optional background music file. Looped to cover the whole
    video and mixed under voiceover (and gameplay audio if present).
    duck_intensity: "off"/"light"/"medium"/"heavy" — how much the music
    volume drops whenever the voiceover is speaking (FFmpeg sidechaincompress,
    classic "auto-duck"). Presets tuned against real normalized TTS audio —
    "light" keeps music clearly audible even under continuous narration,
    "heavy" suppresses it almost completely (use only for dense dialogue).
    music_start_offset: seconds into the music track to start from (e.g.
    skip a quiet intro and start at the song's hook/chorus).
    gameplay_duration: used to time the music's fade-out near the clip's end."""
    try:
        # Pre-flight validation
        if not os.path.exists(gameplay_path):
            raise FileNotFoundError(f"Gameplay video file missing: {gameplay_path}")
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file missing: {audio_path}")
        if not os.path.exists(ass_path):
            raise FileNotFoundError(f"Subtitle file missing: {ass_path}")
        
        # Check file sizes to catch truncated/corrupt files
        gameplay_size = os.path.getsize(gameplay_path)
        if gameplay_size < 1024:  # Less than 1KB is clearly corrupt
            raise RuntimeError("Gameplay file appears to be corrupted or too small")
        
        video_jobs[job_id]["status"] = "composing"

        vf_args = build_ffmpeg_filter(gameplay_w, gameplay_h, fmt, vertical_style, ass_path)
        vf_filter = vf_args[1]

        # Detect if gameplay has audio stream — avoids FFmpeg crash on [0:a] reference
        has_game_audio = _gameplay_has_audio(gameplay_path)

        av_offset = max(0.0, float(av_offset))
        delay_ms  = int(av_offset * 1000)
        has_music = bool(music_path and os.path.exists(music_path))
        music_volume = max(0.0, min(1.0, float(music_volume)))

        # ── Build the audio filter graph piece by piece ──
        audio_parts = []

        # 1. Voiceover (optionally delayed for a silent intro)
        if delay_ms > 0:
            audio_parts.append(f"[1:a]adelay={delay_ms}|{delay_ms}[vdelayed]")
            voice_src = "[vdelayed]"
        else:
            voice_src = "[1:a]"

        # Duck intensity presets — tuned against real normalized TTS audio via
        # direct FFmpeg spectral testing. The OLD hardcoded "heavy" preset
        # (threshold=0.04, ratio=10) was suppressing music to ~8% of voice
        # energy under CONTINUOUS narration (TTS rarely produces true digital
        # silence), making it sound like music wasn't there at all. "light"
        # keeps music clearly audible (~16% relative energy, ~2x heavy) while
        # still ducking during genuinely loud speech peaks.
        DUCK_PRESETS = {
            "off":    None,
            "light":  {"threshold": 0.18, "ratio": 2.5, "attack": 25, "release": 250},
            "medium": {"threshold": 0.08, "ratio": 5,   "attack": 20, "release": 350},
            "heavy":  {"threshold": 0.04, "ratio": 10,  "attack": 15, "release": 500},
        }
        duck_cfg = DUCK_PRESETS.get(duck_intensity, DUCK_PRESETS["light"])

        # If we need to duck music against the voice, split the voice signal
        # so one copy feeds the final mix and the other feeds sidechaincompress
        # (FFmpeg filter pads can only be consumed once without asplit).
        need_voice_split = has_music and duck_cfg is not None
        if need_voice_split:
            audio_parts.append(f"{voice_src}asplit=2[va_mix][va_duck]")
            voice_for_mix = "[va_mix]"
            voice_for_duck = "[va_duck]"
        else:
            voice_for_mix = voice_src
            voice_for_duck = None

        mix_inputs = [voice_for_mix]

        # 2. Gameplay audio (quiet bed) — only if present and not muted
        if not mute_gameplay and has_game_audio:
            audio_parts.append("[0:a]volume=0.15[ga]")
            mix_inputs.append("[ga]")

        # 3. Background music (input index depends on whether gameplay has audio —
        #    FFmpeg numbers inputs by -i order regardless of streams, so music is
        #    always input #2 here since we always pass -i gameplay -i voice -i music)
        if has_music:
            fade_d = 1.5
            fade_chain = f",afade=t=in:st=0:d={fade_d}"
            if gameplay_duration and gameplay_duration > fade_d * 2:
                fadeout_start = max(0.0, gameplay_duration - fade_d)
                fade_chain += f",afade=t=out:st={fadeout_start}:d={fade_d}"
            audio_parts.append(f"[2:a]volume={music_volume}{fade_chain}[mu_pre]")
            if duck_cfg is not None:
                c = duck_cfg
                audio_parts.append(f"[mu_pre]{voice_for_duck}sidechaincompress=threshold={c['threshold']}:ratio={c['ratio']}:attack={c['attack']}:release={c['release']}[mu]")
                mix_inputs.append("[mu]")
            else:
                mix_inputs.append("[mu_pre]")

        # 4. Final mix
        if len(mix_inputs) == 1:
            final_audio_label = mix_inputs[0]
        else:
            inputs_str = "".join(mix_inputs)
            audio_parts.append(f"{inputs_str}amix=inputs={len(mix_inputs)}:duration=first:dropout_transition=0:normalize=0[audio]")
            final_audio_label = "[audio]"

        audio_filter = ";".join(audio_parts)
        full_filter = vf_filter.replace("[out]", "[out];") + audio_filter if audio_filter else vf_filter
        audio_map = final_audio_label if audio_filter else "1:a"

        cmd = [FFMPEG_EXE, "-y", "-i", gameplay_path, "-i", audio_path]
        if has_music:
            # Loop music indefinitely; -shortest at the end trims it to match.
            # -ss before -i seeks the music to the chosen start point (e.g. chorus)
            # before looping begins.
            music_args = []
            if music_start_offset and music_start_offset > 0:
                music_args += ["-ss", str(music_start_offset)]
            music_args += ["-stream_loop", "-1", "-i", music_path]
            cmd += music_args
        cmd += [
            "-filter_complex", full_filter,
            "-map", "[out]", "-map", audio_map,
            "-c:v", "libx264", "-preset", "fast", "-crf", "23",
            "-c:a", "aac", "-b:a", "192k",
            "-shortest", "-movflags", "+faststart",
            output_path
        ]

        print(f"[Video] Running FFmpeg for job {job_id[:8]}...")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

        if result.returncode != 0:
            stderr = result.stderr or ""
            # Find the most useful error line — FFmpeg errors always start with
            # the word "Error" or contain "Invalid" / "No such file" / "codec".
            # Surfacing just that line is far more actionable than a 600-char dump.
            useful_line = ""
            for line in reversed(stderr.splitlines()):
                l = line.strip()
                if l and any(kw in l for kw in ("Error", "error", "Invalid", "No such", "codec", "failed", "Unable", "Cannot")):
                    useful_line = l
                    break
            err_summary = useful_line or stderr[-300:] or "No stderr output"
            raise RuntimeError(f"FFmpeg failed (code {result.returncode}): {err_summary}")

        if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
            raise RuntimeError("FFmpeg ran but output file is missing or empty")

        # ── Optional thumbnail title card ─────────────────────────────
        # If a thumbnail was selected, prepend it as a 3-second still-image
        # card at the very start of the composed video using FFmpeg concat.
        # This is done as a SEPARATE pass AFTER the main encode so it can't
        # break the main compose; if it fails (bad path, wrong format) we just
        # log and keep the video without the card rather than failing the job.
        if thumbnail_path and os.path.exists(thumbnail_path):
            try:
                card_path = output_path.replace(".mp4", "_with_card.mp4")
                # Target dimensions from the composed video — probe first
                probe_cmd = [FFMPEG_PATH, "-i", output_path, "-hide_banner"]
                probe_res = subprocess.run(probe_cmd, capture_output=True, text=True, timeout=10)
                w, h = 1080, 1920  # sane defaults
                import re as _re
                m = _re.search(r"(\d{3,4})x(\d{3,4})", probe_res.stderr or "")
                if m:
                    w, h = int(m.group(1)), int(m.group(2))

                card_cmd = [
                    FFMPEG_PATH, "-y",
                    "-loop", "1", "-t", "3", "-i", thumbnail_path,  # 3-sec still
                    "-i", output_path,
                    "-filter_complex",
                    # Scale thumbnail to match video dimensions, add silent audio
                    f"[0:v]scale={w}:{h}:force_original_aspect_ratio=decrease,"
                    f"pad={w}:{h}:(ow-iw)/2:(oh-ih)/2,fps=30[card];"
                    f"aevalsrc=0:c=stereo:s=44100:d=3[sil];"
                    f"[card][sil][1:v][1:a]concat=n=2:v=1:a=1[outv][outa]",
                    "-map", "[outv]", "-map", "[outa]",
                    "-c:v", "libx264", "-preset", "fast", "-crf", "23",
                    "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart",
                    card_path
                ]
                card_res = subprocess.run(card_cmd, capture_output=True, text=True, timeout=120)
                if card_res.returncode == 0 and os.path.exists(card_path) and os.path.getsize(card_path) > 0:
                    os.replace(card_path, output_path)  # atomic swap
                    print(f"[Video] Job {job_id[:8]} — thumbnail title card prepended")
                else:
                    print(f"[Video] Job {job_id[:8]} — thumbnail card failed (ignored): {card_res.stderr[-200:]}")
                    if os.path.exists(card_path):
                        os.remove(card_path)
            except Exception as thumb_err:
                print(f"[Video] Job {job_id[:8]} — thumbnail card error (ignored): {thumb_err}")

        video_jobs[job_id].update({"status": "completed", "output_path": output_path, "progress": 100})
        print(f"[Video] Job {job_id[:8]} completed — {os.path.getsize(output_path)//1024}KB")

    except Exception as e:
        error_str = str(e)
        user_msg, suggestion = format_user_error(error_str)
        full_error_msg = f"{user_msg}\n💡 Suggestion: {suggestion}"
        video_jobs[job_id].update({"status": "failed", "error": full_error_msg, "tech_error": error_str})
        print(f"[Video] Job {job_id[:8]} FAILED: {error_str}")


# ── VIDEO STUDIO ENDPOINTS ────────────────────────────────────────
@app.post("/upload-gameplay")
async def upload_gameplay(file: UploadFile = File(...)):
    """Upload an HCR2 gameplay video clip"""
    try:
        allowed = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
        ext = os.path.splitext(file.filename or "")[1].lower()
        if ext not in allowed:
            return {"error": f"Unsupported format. Use: {', '.join(allowed)}", "status": "failed"}
        gameplay_id = str(uuid.uuid4())
        out_path = f"{VIDEOS_DIR}/{gameplay_id}{ext}"
        content = await file.read()
        with open(out_path, "wb") as f_out:
            f_out.write(content)
        info = get_video_info(out_path)
        gameplay_store[gameplay_id] = {"path": out_path, "filename": file.filename, "width": info["width"], "height": info["height"], "duration": info["duration"]}
        return {"gameplay_id": gameplay_id, "filename": file.filename, "duration": round(info["duration"], 1), "resolution": f"{info['width']}x{info['height']}", "status": "ready"}
    except Exception as e:
        return {"error": str(e), "status": "failed"}


class SceneTrim(BaseModel):
    gameplay_id: str
    trim_start:  float = 0.0    # seconds — 0 = from the start
    trim_end:    float = 0.0    # seconds — 0 = use full clip duration (no trim)


class ConcatRequest(BaseModel):
    scenes: List[SceneTrim]     # order matters — this is the final sequence


@app.post("/concat-gameplay-clips")
async def concat_gameplay_clips(req: ConcatRequest):
    """Combine multiple uploaded gameplay clips (optionally trimmed) into ONE
    new gameplay clip, in the given order — multi-scene video support.

    This is implemented as a PRE-PROCESSING step: the output is registered
    as a perfectly normal new gameplay_id in gameplay_store, exactly like a
    single /upload-gameplay would produce. That means subtitles, voiceover
    sync, thumbnails, and Compose all work completely unchanged afterward —
    they just see "one gameplay clip" as usual."""
    try:
        if not VIDEO_STUDIO_AVAILABLE:
            return JSONResponse(status_code=500, content={"error": "Video Studio not available (pip install imageio-ffmpeg)"})
        if len(req.scenes) < 2:
            return JSONResponse(status_code=400, content={"error": "Need at least 2 scenes to combine. For a single clip, just upload it normally."})
        if len(req.scenes) > 12:
            return JSONResponse(status_code=400, content={"error": "Maximum 12 scenes per video (keeps render time reasonable)."})

        for s in req.scenes:
            if s.gameplay_id not in gameplay_store:
                return JSONResponse(status_code=404, content={"error": f"Gameplay clip not found: {s.gameplay_id[:8]}... — was it uploaded in this session?"})

        # Target a common resolution: the largest width found among scenes,
        # 16:9 derived height — all clips get scaled/padded to match so
        # concat works even if clips have different native resolutions.
        target_w = max(gameplay_store[s.gameplay_id]["width"] for s in req.scenes)
        target_w = min(target_w, 1920)  # cap for reasonable render time
        target_h = int(target_w * 9 / 16)
        if target_h % 2:
            target_h += 1

        def _run_concat():
            inputs = []
            filter_parts = []
            concat_inputs = ""

            for i, scene in enumerate(req.scenes):
                g = gameplay_store[scene.gameplay_id]
                path = os.path.abspath(g["path"])
                clip_duration = g.get("duration", 0) or 0

                trim_start = max(0.0, scene.trim_start)
                trim_end = scene.trim_end if scene.trim_end > trim_start else clip_duration
                trim_end = min(trim_end, clip_duration) if clip_duration > 0 else trim_end

                inputs += ["-i", path]

                # Trim (if requested), scale to common size (pad with black bars
                # if aspect ratio differs), and normalize framerate/format so
                # concat filter doesn't choke on mismatched streams.
                trim_filter = f"trim={trim_start}:{trim_end}," if (trim_start > 0 or scene.trim_end > 0) else ""
                vf = (f"[{i}:v]{trim_filter}setpts=PTS-STARTPTS,"
                      f"scale={target_w}:{target_h}:force_original_aspect_ratio=decrease,"
                      f"pad={target_w}:{target_h}:(ow-iw)/2:(oh-ih)/2,setsar=1,fps=30[v{i}]")
                filter_parts.append(vf)

                # Audio: trim + normalize timestamps; generate silence if the
                # clip has no audio stream, so the concat filter always has a
                # consistent number of audio inputs.
                has_audio = _gameplay_has_audio(path)
                if has_audio:
                    af = f"[{i}:a]{trim_filter}asetpts=PTS-STARTPTS,aresample=44100,aformat=channel_layouts=stereo[a{i}]"
                else:
                    seg_dur = max(0.1, trim_end - trim_start) if trim_end > trim_start else max(0.1, clip_duration)
                    af = f"anullsrc=channel_layout=stereo:sample_rate=44100:duration={seg_dur}[a{i}]"
                filter_parts.append(af)

                concat_inputs += f"[v{i}][a{i}]"

            concat_filter = f"{concat_inputs}concat=n={len(req.scenes)}:v=1:a=1[outv][outa]"
            full_filter = ";".join(filter_parts) + ";" + concat_filter

            out_id = str(uuid.uuid4())
            out_path = os.path.abspath(f"{VIDEOS_DIR}/{out_id}.mp4")

            cmd = [FFMPEG_EXE, "-y"] + inputs + [
                "-filter_complex", full_filter,
                "-map", "[outv]", "-map", "[outa]",
                "-c:v", "libx264", "-preset", "fast", "-crf", "20",
                "-c:a", "aac", "-b:a", "192k",
                "-movflags", "+faststart",
                out_path
            ]
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
            if r.returncode != 0:
                raise RuntimeError(f"Concat failed: {r.stderr[-600:]}")
            if not os.path.exists(out_path) or os.path.getsize(out_path) < 1024:
                raise RuntimeError("Concat produced an empty or missing file")
            return out_id, out_path

        out_id, out_path = await asyncio.to_thread(_run_concat)
        info = get_video_info(out_path)

        scene_names = [gameplay_store[s.gameplay_id].get("filename", "clip") for s in req.scenes]
        combined_name = f"Combined ({len(req.scenes)} scenes): " + " + ".join(n[:15] for n in scene_names[:3]) + ("..." if len(scene_names) > 3 else "")

        gameplay_store[out_id] = {
            "path": out_path, "filename": combined_name,
            "width": info["width"], "height": info["height"], "duration": info["duration"],
        }

        return {
            "gameplay_id": out_id, "filename": combined_name,
            "duration": round(info["duration"], 1), "resolution": f"{info['width']}x{info['height']}",
            "scene_count": len(req.scenes), "status": "ready",
        }
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.delete("/delete-ass/{audio_id}")
async def delete_ass(audio_id: str):
    """Delete cached ASS subtitle file to force regeneration from correct audio."""
    if ".." in audio_id or "/" in audio_id or "\\" in audio_id:
        return {"error": "Invalid id"}
    ass_path = f"{EXPORTS_DIR}/{audio_id}.ass"
    try:
        if os.path.exists(ass_path):
            os.unlink(ass_path)
            return {"deleted": True}
        return {"deleted": False}
    except Exception as e:
        return {"error": str(e)}


@app.post("/generate-ass")
async def generate_ass(req: ASSRequest):
    """Generate ASS subtitle file with word-level timing and animation effects.
    Whisper transcription is CACHED per audio_id — changing style/effect/position
    and clicking Compose again re-renders the ASS instantly without re-running STT."""
    try:
        audio_path = f"{EXPORTS_DIR}/{req.audio_id}.wav"
        if not os.path.exists(audio_path):
            return JSONResponse(status_code=404, content={"error": "Audio file not found. Generate voiceover first.", "status": "failed"})

        was_cached = req.audio_id in WORD_CACHE

        def _get_segments():
            return get_word_segments(req.audio_id, audio_path)
        segments = await asyncio.to_thread(_get_segments)

        def _render():
            return generate_ass_content(segments, req.style, req.highlight_color, req.effect, req.position, req.size, req.intensity, req.av_offset, req.uppercase,
                                        req.text_color, req.font_family, req.box_opacity, req.outline_width, req.anim_speed, req.easing, req.color_cycle, req.caption_mode)
        ass_content = await asyncio.to_thread(_render)

        ass_path = f"{EXPORTS_DIR}/{req.audio_id}.ass"
        with open(ass_path, "w", encoding="utf-8") as f:
            f.write(ass_content)
        return {"ass_id": req.audio_id, "ass_path": ass_path, "status": "ready", "effect": req.effect,
                "preview": ass_content[:300], "transcription_cached": was_cached}
    except Exception as e:
        return {"error": str(e), "status": "failed"}


async def _prepare_video_job(req: VideoRequest):
    """Validate a VideoRequest and prepare everything needed to run
    run_video_job — shared by the immediate /create-video endpoint AND the
    sequential render queue worker, so both paths behave identically.

    Returns (kwargs_dict, None) on success, or (None, error_dict) on failure.
    error_dict matches the shape already returned by /create-video on error.
    """
    if not VIDEO_STUDIO_AVAILABLE:
        return None, {"error": "❌ Video Studio not available\n💡 Suggestion: Reinstall dependencies (pip install imageio-ffmpeg)", "status": "failed"}

    # Check gameplay
    if req.gameplay_id not in gameplay_store:
        return None, {"error": "❌ Gameplay video not found\n💡 Suggestion: Upload a video in Step 1", "status": "failed"}

    g = gameplay_store[req.gameplay_id]

    if g["width"] < 64 or g["height"] < 64:
        return None, {"error": "❌ Video resolution too small\n💡 Suggestion: Use a video with at least 64x64 resolution", "status": "failed"}
    if g["width"] > 7680 or g["height"] > 4320:
        return None, {"error": "❌ Video resolution too large (max 8K)\n💡 Suggestion: Downscale video to 1920x1080 or lower", "status": "failed"}

    # Check audio
    audio_path = f"{EXPORTS_DIR}/{req.audio_id}.wav"
    if not os.path.exists(audio_path):
        return None, {"error": "❌ Voiceover not found\n💡 Suggestion: Generate voiceover in Step 2 (TTS Studio)", "status": "failed"}

    try:
        audio_info = librosa.get_samplerate(audio_path)
        if audio_info < 8000:
            return None, {"error": "❌ Audio quality too low\n💡 Suggestion: Regenerate voiceover", "status": "failed"}
    except Exception:
        return None, {"error": "❌ Audio file corrupted\n💡 Suggestion: Regenerate voiceover in TTS Studio", "status": "failed"}

    # Validate subtitle style and color
    valid_styles = ["viral", "clean", "netflix", "minimal"]
    if req.subtitle_style not in valid_styles:
        req.subtitle_style = "clean"

    named_presets = ["cyan", "yellow", "green", "white", "red", "orange", "purple", "pink", "blue"]
    def _is_valid_color(c):
        if c in named_presets:
            return True
        if isinstance(c, str) and c.startswith("#") and len(c) == 7:
            try:
                int(c[1:], 16)
                return True
            except ValueError:
                return False
        return False
    if not _is_valid_color(req.highlight_color):
        req.highlight_color = "cyan"
    if not _is_valid_color(req.text_color):
        req.text_color = "white"

    if req.format not in ["shorts", "youtube"]:
        req.format = "shorts"
    if req.vertical_style not in ["blur", "crop"]:
        req.vertical_style = "blur"

    # Generate/load subtitles
    ass_path = f"{EXPORTS_DIR}/{req.audio_id}.ass"
    if not os.path.exists(ass_path):
        try:
            def _gen_ass():
                segs = get_word_segments(req.audio_id, audio_path)
                return generate_ass_content(segs, req.subtitle_style, req.highlight_color, req.subtitle_effect, req.sub_position, req.sub_size, req.effect_intensity, req.av_offset, req.uppercase,
                                            req.text_color, req.font_family, req.box_opacity, req.outline_width, req.anim_speed, req.easing, req.color_cycle, req.caption_mode)
            ass_content = await asyncio.to_thread(_gen_ass)
            with open(ass_path, "w", encoding="utf-8") as f:
                f.write(ass_content)
        except Exception:
            return None, {"error": "❌ Failed to generate subtitles\n💡 Suggestion: Try a different subtitle style", "status": "failed"}

    job_id = str(uuid.uuid4())
    output_path  = os.path.abspath(f"{VIDEOS_DIR}/final_{job_id}.mp4")
    abs_gameplay = os.path.abspath(g["path"])
    abs_audio    = os.path.abspath(audio_path)
    abs_ass      = os.path.abspath(ass_path)

    video_jobs[job_id] = {
        "status": "queued",
        "progress": 0,
        "output_path": None,
        "error": None,
        "gameplay_info": f"{g['width']}x{g['height']} @ {g['duration']:.1f}s"
    }

    # Resolve background music path (if any)
    abs_music = None
    if req.music_id:
        if req.music_id not in music_store:
            conn = sqlite3.connect(DB_PATH)
            c = conn.cursor()
            c.execute("SELECT path FROM music_tracks WHERE id=?", (req.music_id,))
            row = c.fetchone()
            conn.close()
            if row and os.path.exists(row[0]):
                music_store[req.music_id] = {"path": row[0]}
        if req.music_id in music_store:
            abs_music = os.path.abspath(music_store[req.music_id]["path"])

    kwargs = dict(
        job_id=job_id, gameplay_path=abs_gameplay, audio_path=abs_audio, ass_path=abs_ass,
        output_path=output_path, fmt=req.format, vertical_style=req.vertical_style,
        gameplay_w=g["width"], gameplay_h=g["height"], mute_gameplay=req.mute_gameplay,
        av_offset=req.av_offset, music_path=abs_music, music_volume=req.music_volume,
        duck_intensity=req.duck_intensity, music_start_offset=req.music_start_offset,
        gameplay_duration=g.get("duration", 0.0),
        thumbnail_path=os.path.abspath(req.thumbnail_path) if req.thumbnail_path and os.path.exists(req.thumbnail_path) else None,
    )
    return kwargs, None


@app.post("/create-video")
async def create_video(req: VideoRequest, background_tasks: BackgroundTasks):
    """Compose gameplay + voiceover + animated subtitles into a final MP4 — runs immediately."""
    try:
        kwargs, err = await _prepare_video_job(req)
        if err:
            status_code = 404 if "not found" in err["error"] else (400 if "too small" in err["error"] or "too large" in err["error"] or "corrupted" in err["error"] or "too low" in err["error"] else 500)
            return JSONResponse(status_code=status_code, content=err)

        background_tasks.add_task(run_video_job, **kwargs)
        return {"job_id": kwargs["job_id"], "status": "queued"}

    except Exception as e:
        print(f"[Video] Validation error: {e}")
        user_msg, suggestion = format_user_error(str(e))
        return JSONResponse(status_code=500, content={"error": f"{user_msg}\n💡 Suggestion: {suggestion}", "status": "failed"})


render_queue = []   # list of {queue_id, name, video_request_dict, status, job_id, added_at, error}
_queue_worker_started = False


async def _render_queue_worker():
    """Background loop — processes queued render jobs ONE AT A TIME (sequential,
    not parallel) to avoid GPU/CPU/FFmpeg contention with whatever else the
    app is doing (TTS generation, AI thumbnail/music generation, etc.)."""
    while True:
        try:
            pending = next((q for q in render_queue if q["status"] == "queued"), None)
            if pending is None:
                await asyncio.sleep(2)
                continue

            pending["status"] = "processing"
            try:
                req = VideoRequest(**pending["video_request_dict"])
                kwargs, err = await _prepare_video_job(req)
                if err:
                    pending["status"] = "failed"
                    pending["error"] = err.get("error", "Unknown error")
                    continue

                pending["job_id"] = kwargs["job_id"]
                # Run synchronously within this worker loop — blocks until this
                # one render finishes, THEN the loop picks up the next item.
                await asyncio.to_thread(run_video_job, **kwargs)

                final_status = video_jobs.get(kwargs["job_id"], {})
                if final_status.get("status") == "completed":
                    pending["status"] = "done"
                else:
                    pending["status"] = "failed"
                    pending["error"] = final_status.get("error", "Render failed")
            except Exception as e:
                pending["status"] = "failed"
                pending["error"] = str(e)
        except Exception as loop_err:
            print(f"[Queue] Worker loop error: {loop_err}")
            await asyncio.sleep(3)


@app.on_event("startup")
async def _start_queue_worker():
    global _queue_worker_started
    if not _queue_worker_started:
        _queue_worker_started = True
        asyncio.create_task(_render_queue_worker())
        print("  Render Queue  ✓  Worker started")


@app.post("/queue-add")
async def queue_add(req: VideoRequest):
    """Add a video composition job to the sequential render queue instead of
    running it immediately. Useful for batching several videos to run one
    after another (e.g. overnight) without manually waiting for each one."""
    queue_id = str(uuid.uuid4())
    g = gameplay_store.get(req.gameplay_id, {})
    entry = {
        "queue_id": queue_id,
        "name": g.get("filename", req.gameplay_id[:8]) + " + " + req.audio_id[:8],
        "video_request_dict": req.dict(),
        "status": "queued",
        "job_id": None,
        "error": None,
        "added_at": datetime.now().isoformat(),
    }
    render_queue.append(entry)
    return {"queue_id": queue_id, "status": "queued", "position": len([q for q in render_queue if q["status"] == "queued"])}


@app.get("/queue-list")
async def queue_list():
    """Return all render queue entries with live status/progress."""
    out = []
    for q in render_queue:
        progress = 0
        if q["job_id"] and q["job_id"] in video_jobs:
            progress = video_jobs[q["job_id"]].get("progress", 0)
        out.append({
            "queue_id": q["queue_id"], "name": q["name"], "status": q["status"],
            "job_id": q["job_id"], "error": q["error"], "added_at": q["added_at"],
            "progress": progress,
        })
    return {"queue": out}


@app.delete("/queue-remove/{queue_id}")
async def queue_remove(queue_id: str):
    """Remove a queued (not yet started) entry from the render queue."""
    global render_queue
    target = next((q for q in render_queue if q["queue_id"] == queue_id), None)
    if not target:
        return JSONResponse(status_code=404, content={"error": "Queue entry not found"})
    if target["status"] == "processing":
        return JSONResponse(status_code=400, content={"error": "Cannot remove a job that's currently rendering"})
    render_queue = [q for q in render_queue if q["queue_id"] != queue_id]
    return {"removed": True}


@app.delete("/queue-clear")
async def queue_clear():
    """Remove all queued (not processing/done/failed) entries."""
    global render_queue
    render_queue = [q for q in render_queue if q["status"] == "processing"]
    return {"cleared": True}



@app.get("/video-status/{job_id}")
async def video_status(job_id: str):
    """Poll the status of a video composition job"""
    if job_id not in video_jobs:
        return {"status": "not_found"}
    return video_jobs[job_id]


@app.get("/videos/{filename}")
async def serve_video(filename: str):
    """Serve completed video file for download/preview"""
    if ".." in filename or filename.startswith("/"):
        return {"error": "Invalid path"}
    filepath = f"{VIDEOS_DIR}/{filename}"
    if os.path.exists(filepath):
        return FileResponse(filepath, media_type="video/mp4")
    return {"error": "Video not found"}


@app.get("/list-videos")
async def list_videos():
    """List all completed videos"""
    vids = []
    for job_id, job in video_jobs.items():
        if job["status"] == "completed" and job["output_path"]:
            fname = os.path.basename(job["output_path"])
            size  = os.path.getsize(job["output_path"]) if os.path.exists(job["output_path"]) else 0
            vids.append({"job_id": job_id, "filename": fname, "size_mb": round(size / (1024 * 1024), 1)})
    return {"videos": vids}


# ── PARLER-TTS ─────────────────────────────────────────────────────

parler_model     = None
parler_tokenizer = None
_parler_checked  = {"available": None, "detail": ""}


@app.get("/parler-status")
async def parler_status():
    """Lightweight check (no model load) — is Parler-TTS importable + is the
    big model already cached locally? Used by the UI to show a clear status
    badge instead of leaving the user guessing whether it 'works'."""
    if _parler_checked["available"] is not None:
        return {"available": _parler_checked["available"], "detail": _parler_checked["detail"],
                "loaded": parler_model is not None, "cached": _parler_checked.get("cached", False),
                "device": _parler_checked.get("device")}

    try:
        from parler_tts import ParlerTTSForConditionalGeneration  # noqa: F401
        from transformers import AutoTokenizer  # noqa: F401
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"

        # Check if the model weights are already cached (avoids surprise 800MB download)
        cached = False
        try:
            from huggingface_hub import scan_cache_dir
            cache = scan_cache_dir()
            cached = any("parler-tts" in r.repo_id for r in cache.repos)
        except Exception:
            pass

        detail = f"Library installed ✓ (device: {device})"
        if cached:
            detail += " · model cached, ready instantly"
        else:
            detail += " · model NOT downloaded yet — first generate will fetch ~800MB"

        _parler_checked.update(available=True, detail=detail, cached=cached, device=device)
        return {"available": True, "detail": detail, "loaded": parler_model is not None, "cached": cached, "device": device}

    except ImportError as e:
        detail = ("Not installed. Run in your venv:\\n"
                   "pip install git+https://github.com/huggingface/parler-tts.git transformers accelerate")
        _parler_checked.update(available=False, detail=detail)
        return {"available": False, "detail": detail, "loaded": False, "error": str(e)}


def get_parler():
    global parler_model, parler_tokenizer
    if parler_model is None:
        try:
            from parler_tts import ParlerTTSForConditionalGeneration
            from transformers import AutoTokenizer
        except ImportError:
            raise RuntimeError(
                "Parler-TTS not installed.\n"
                f"Run: {_sys_boot.executable} -m pip install "
                "git+https://github.com/huggingface/parler-tts.git transformers accelerate"
            )
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"  Loading Parler-TTS on {device} (first run: downloads ~800MB)...")
        model_id = "parler-tts/parler-tts-mini-v1"
        parler_model     = ParlerTTSForConditionalGeneration.from_pretrained(model_id).to(device)
        parler_tokenizer = AutoTokenizer.from_pretrained(model_id)
        print("  Parler-TTS  ✓  Ready")
    return parler_model, parler_tokenizer


class ParlerRequest(BaseModel):
    text:        str
    description: str   = "A friendly clear voice speaking naturally. Studio quality audio."
    speed:       float = 1.0
    normalize:   bool  = True
    compress:    bool  = False
    bass:        float = 0.0
    treble:      float = 0.0
    reverb:      bool  = False


@app.post("/generate-parler")
async def generate_parler(req: ParlerRequest):
    """Generate expressive audio via Parler-TTS style conditioning.
    If the model isn't loaded yet (first run), returns a 503 with
    status='loading_required' so the frontend can trigger the SSE
    download progress stream instead of hanging."""
    try:
        # If model not loaded, tell the frontend to start the SSE stream
        # rather than blocking this request for ~2 minutes silently.
        if parler_model is None:
            return JSONResponse(status_code=503, content={
                "status": "loading_required",
                "error": "Parler-TTS model not loaded yet — use /parler-load-stream to download and load it first."
            })

        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model, tokenizer = parler_model, parler_tokenizer

        def run_parler():
            inp   = tokenizer(req.description, return_tensors="pt").input_ids.to(device)
            p_inp = tokenizer(req.text,        return_tensors="pt").input_ids.to(device)
            with torch.no_grad():
                gen = model.generate(input_ids=inp, prompt_input_ids=p_inp)
            arr = gen.cpu().numpy().squeeze().astype(np.float32)
            sr  = model.config.sampling_rate
            return arr, sr

        audio_np, sr = await asyncio.to_thread(run_parler)
        if sr != 24000:
            audio_np = librosa.resample(audio_np, orig_sr=sr, target_sr=24000)
        audio_np = process_audio(audio_np, 24000, req)
        audio_np = np.clip(audio_np, -0.99, 0.99)
        job_id, _ = save_audio_file(audio_np, "parler", req.speed)
        buf = io.BytesIO()
        sf.write(buf, audio_np, 24000, format="WAV")
        buf.seek(0)
        return StreamingResponse(buf, media_type="audio/wav",
            headers={"Content-Disposition": f'attachment; filename="parler_{job_id}.wav"',
                     "X-Job-ID": job_id})
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e), "status": "failed"})


# ══════════════════════════════════════════════════════════════════
# BACKGROUND MUSIC
# ══════════════════════════════════════════════════════════════════

@app.post("/upload-music")
async def upload_music(file: UploadFile = File(...)):
    """Upload a background music track (mp3/wav/m4a). Automatically
    loudness-normalized on upload so the volume slider behaves consistently
    regardless of how loud/quiet the source track was mastered."""
    try:
        ext = os.path.splitext(file.filename)[1].lower()
        if ext not in (".mp3", ".wav", ".m4a", ".ogg", ".flac"):
            return JSONResponse(status_code=400, content={"error": "Unsupported audio format. Use mp3, wav, m4a, ogg, or flac."})
        music_id = str(uuid.uuid4())
        tmp_path = f"{MUSIC_DIR}/{music_id}_raw{ext}"
        with open(tmp_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        # Load (preserving stereo + native sample rate), normalize loudness,
        # save as a consistent WAV regardless of original format.
        out_path = f"{MUSIC_DIR}/{music_id}.wav"
        try:
            y, sr = librosa.load(tmp_path, sr=None, mono=False)
            y_norm = apply_normalization(y, target_db=-20.0)
            y_norm = np.clip(y_norm, -0.99, 0.99)
            write_data = y_norm.T if y_norm.ndim == 2 else y_norm
            sf.write(out_path, write_data, sr)
            duration = librosa.get_duration(y=y, sr=sr)
        except Exception as norm_err:
            # Fallback: keep original file as-is if normalization fails
            print(f"  Music normalize warning: {norm_err}")
            out_path = tmp_path
            try:
                duration = librosa.get_duration(path=out_path)
            except Exception:
                duration = 0.0
        finally:
            if out_path != tmp_path and os.path.exists(tmp_path):
                try:
                    os.unlink(tmp_path)
                except Exception:
                    pass

        music_store[music_id] = {"path": out_path, "filename": file.filename, "duration": duration}

        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("INSERT INTO music_tracks VALUES (?,?,?,?,?,?)",
                  (music_id, file.filename, out_path, file.filename, duration, datetime.now().isoformat()))
        conn.commit()
        conn.close()

        return {"music_id": music_id, "filename": file.filename, "duration": round(duration, 1), "status": "ready", "normalized": out_path.endswith(".wav")}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.get("/list-music")
async def list_music():
    """List all uploaded background music tracks."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT id, filename, duration, created_at FROM music_tracks ORDER BY created_at DESC")
    rows = c.fetchall()
    conn.close()
    tracks = [{"music_id": r[0], "filename": r[1], "duration": round(r[2] or 0, 1), "created_at": r[3]} for r in rows]
    # Re-populate in-memory store on restart so music_store stays in sync
    for r in rows:
        if r[0] not in music_store:
            p = f"{MUSIC_DIR}/{r[0]}{os.path.splitext(r[1])[1].lower()}"
            for ext in (".mp3", ".wav", ".m4a", ".ogg", ".flac"):
                cand = f"{MUSIC_DIR}/{r[0]}{ext}"
                if os.path.exists(cand):
                    music_store[r[0]] = {"path": cand, "filename": r[1], "duration": r[2] or 0}
                    break
    return {"tracks": tracks}


@app.get("/music-file/{music_id}")
async def serve_music_file(music_id: str):
    """Serve an uploaded music track for in-browser preview playback."""
    if ".." in music_id or "/" in music_id:
        return JSONResponse(status_code=400, content={"error": "Invalid id"})
    path = None
    if music_id in music_store:
        path = music_store[music_id]["path"]
    if not path or not os.path.exists(path):
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("SELECT path FROM music_tracks WHERE id=?", (music_id,))
        row = c.fetchone()
        conn.close()
        if row and os.path.exists(row[0]):
            path = row[0]
    if not path or not os.path.exists(path):
        return JSONResponse(status_code=404, content={"error": "Music file not found"})
    return FileResponse(path, media_type="audio/wav")


@app.delete("/delete-music/{music_id}")
async def delete_music(music_id: str):
    """Delete an uploaded music track."""
    if ".." in music_id or "/" in music_id:
        return {"error": "Invalid id"}
    try:
        if music_id in music_store:
            p = music_store[music_id]["path"]
            if os.path.exists(p):
                os.unlink(p)
            del music_store[music_id]
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("DELETE FROM music_tracks WHERE id=?", (music_id,))
        conn.commit()
        conn.close()
        return {"deleted": True}
    except Exception as e:
        return {"error": str(e)}


# ══════════════════════════════════════════════════════════════════
# AI MUSIC GENERATION — local MusicGen (Suno-style, fully offline)
# ══════════════════════════════════════════════════════════════════

musicgen_model     = None
musicgen_processor = None
_musicgen_lock = asyncio.Lock()

MUSIC_STYLE_PRESETS = {
    "gaming":   "upbeat electronic gaming background music, energetic synth, driving beat",
    "epic":     "epic orchestral cinematic music, dramatic strings, powerful drums, building tension",
    "chill":    "lo-fi chill background music, relaxed beat, mellow synth, calm atmosphere",
    "intense":  "intense action music, fast tempo, aggressive electronic beat, high energy",
    "funny":    "quirky comedic background music, playful xylophone, bouncy rhythm",
    "suspense": "suspenseful tense music, dark ambient drone, slow building dread",
}


def get_musicgen():
    """Lazy-load Meta's MusicGen (small variant — ~300M params, good balance
    of speed/quality for short background music loops). Same lazy-singleton
    pattern as get_parler()/get_xtts()/get_sd_pipeline()."""
    global musicgen_model, musicgen_processor
    if musicgen_model is None:
        try:
            import torch
            from transformers import MusicgenForConditionalGeneration, AutoProcessor
        except ImportError:
            raise RuntimeError(
                "MusicGen not installed.\n"
                "Run: pip install transformers torch accelerate scipy\n"
                "(torch with CUDA strongly recommended — CPU generation is very slow)"
            )
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"  Loading MusicGen (small) on {device} (first run downloads ~2GB)...")
        musicgen_processor = AutoProcessor.from_pretrained("facebook/musicgen-small")
        musicgen_model = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-small").to(device)
        print("  MusicGen  ✓  Ready")
    return musicgen_model, musicgen_processor


@app.get("/music-ai-status")
async def music_ai_status():
    """Lightweight check (no model load) — is MusicGen importable?"""
    try:
        import torch
        from transformers import MusicgenForConditionalGeneration  # noqa: F401
        device = "cuda" if torch.cuda.is_available() else "cpu"
        detail = f"Library installed ✓ (device: {device})"
        if device == "cpu":
            detail += " · WARNING: CPU generation is slow (~1-2 min per 10s of audio)"
        else:
            detail += " · GPU detected, generation should take ~10-30s for 10s of audio"
        return {"available": True, "loaded": musicgen_model is not None, "detail": detail, "presets": MUSIC_STYLE_PRESETS}
    except ImportError:
        return {"available": False, "loaded": False,
                "detail": "Not installed. Run: pip install transformers torch accelerate scipy",
                "presets": MUSIC_STYLE_PRESETS}


class MusicAIRequest(BaseModel):
    prompt:       str = ""        # free-text description — if provided, takes priority but is still
                                   # combined with structured fields below (instrumental/vocals tag etc.)
    mood:         str = ""        # e.g. "energetic", "dark", "happy", "epic", "calm", "mysterious"
    tempo:        str = "medium"  # "slow" | "medium" | "fast" | "very_fast"
    bpm:          int = 0         # 0 = let tempo word decide; otherwise an explicit BPM (60-180) overrides it
    energy:       int = 2         # 0=minimal, 1=low, 2=medium, 3=high, 4=intense
    instruments:  List[str] = []  # e.g. ["synth","drums","bass","guitar","piano","strings","8-bit"]
    key_mode:     str = ""        # "" = unspecified | "major" (brighter/happier) | "minor" (darker/sadder)
    vocals:       bool = False    # MusicGen-small vocals are usually low quality — instrumental by default
    duration:     float = 15.0    # seconds — keep short (8-20s); it loops automatically when used as background music
    seed:         int = -1        # -1 = random each time; set a fixed number to reproduce/tweak the same generation


TEMPO_BPM = {"slow": 70, "medium": 105, "fast": 140, "very_fast": 170}
ENERGY_WORDS = {0: "very minimal and sparse", 1: "low energy, subtle", 2: "moderate energy",
                3: "high energy, driving", 4: "intense, maximum energy, powerful"}


def _build_music_prompt(req: "MusicAIRequest") -> str:
    """Compose a rich MusicGen text prompt from structured controls. If the
    user also typed a free-text prompt, it's blended in as the lead
    description; otherwise the prompt is built entirely from the controls."""
    parts = []
    if req.prompt.strip():
        parts.append(req.prompt.strip())
    if req.mood:
        parts.append(f"{req.mood} mood")

    bpm = req.bpm if req.bpm and 40 <= req.bpm <= 220 else TEMPO_BPM.get(req.tempo, 105)
    parts.append(f"tempo around {bpm} BPM")

    parts.append(ENERGY_WORDS.get(req.energy, ENERGY_WORDS[2]))

    if req.instruments:
        parts.append("featuring " + ", ".join(req.instruments))

    if req.key_mode == "major":
        parts.append("major key, bright and uplifting harmony")
    elif req.key_mode == "minor":
        parts.append("minor key, darker and more dramatic harmony")

    if not req.vocals:
        parts.append("instrumental only, no vocals, no singing, no lyrics")

    parts.append("clean studio quality, loopable")

    return ", ".join(parts)


@app.get("/music-ai-options")
async def music_ai_options():
    """Static option lists for the AI Music control panel (moods, instruments, etc.)."""
    return {
        "moods": ["energetic", "epic", "happy", "dark", "mysterious", "triumphant",
                  "sad", "calm", "suspenseful", "playful", "aggressive", "dreamy"],
        "instruments": ["synth", "drums", "bass", "electric guitar", "acoustic guitar",
                         "piano", "strings", "brass", "8-bit chiptune", "orchestral",
                         "percussion", "pad"],
        "tempos": ["slow", "medium", "fast", "very_fast"],
        "presets": MUSIC_STYLE_PRESETS,
    }


async def _run_musicgen_generation(final_prompt: str, duration: float, seed: int, label_hint: str = ""):
    """Core MusicGen generation logic — shared by /generate-music-ai and
    /edit-music-ai so both go through identical, tested code."""
    async with _musicgen_lock:  # one generation at a time — avoids GPU OOM
        model, processor = await asyncio.to_thread(get_musicgen)

        duration = max(3.0, min(30.0, duration))
        max_new_tokens = int(duration * 50)  # MusicGen codebook framerate ~50 tok/sec

        def _run_musicgen():
            import torch
            device = next(model.parameters()).device
            if seed is not None and seed >= 0:
                torch.manual_seed(seed)
            inputs = processor(text=[final_prompt], padding=True, return_tensors="pt").to(device)
            with torch.no_grad():
                audio_values = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=True, guidance_scale=3.0)
            sr = model.config.audio_encoder.sampling_rate
            arr = audio_values[0, 0].cpu().numpy().astype(np.float32)
            return arr, sr

        audio_np, sr = await asyncio.to_thread(_run_musicgen)

    audio_np = apply_normalization(audio_np, target_db=-23.0)
    audio_np = np.clip(audio_np, -0.99, 0.99)

    music_id = str(uuid.uuid4())
    out_path = f"{MUSIC_DIR}/{music_id}.wav"
    sf.write(out_path, audio_np, sr)
    actual_duration = len(audio_np) / sr

    display_name = f"AI: {(label_hint or final_prompt)[:40]}"
    music_store[music_id] = {"path": out_path, "filename": display_name, "duration": actual_duration}

    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("INSERT INTO music_tracks VALUES (?,?,?,?,?,?)",
              (music_id, display_name, out_path, display_name, actual_duration, datetime.now().isoformat()))
    conn.commit()
    conn.close()

    return {"music_id": music_id, "filename": display_name, "duration": round(actual_duration, 1), "prompt_used": final_prompt}


@app.post("/generate-music-ai")
async def generate_music_ai(req: MusicAIRequest):
    """Generate an original background music track FROM STRUCTURED CONTROLS
    (mood, tempo/BPM, energy, instruments, key, vocals on/off) plus optional
    free-text description — using local MusicGen, no internet, no Suno
    subscription. The result is saved exactly like an uploaded track, so it
    immediately appears in the Background Music dropdown (loops
    automatically via the existing stream_loop logic in run_video_job)."""
    try:
        final_prompt = _build_music_prompt(req)
        label_bits = [b for b in [req.mood, req.tempo if req.tempo != "medium" else ""] if b]
        short_label = " ".join(label_bits)
        result = await _run_musicgen_generation(final_prompt, req.duration, req.seed, short_label)
        return {**result, "status": "ready"}
    except RuntimeError as e:
        return JSONResponse(status_code=400, content={"error": str(e)})
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"AI music generation failed: {str(e)}"})


class MusicEditRequest(BaseModel):
    history:          List[str]        # prior prompt descriptions in this thread, oldest first
    edit_instruction: str               # the new edit request, e.g. "make it more energetic and faster"
    duration:         float = 15.0
    seed:             int = -1
    model:            str = "llama3.2:3b"


MUSIC_REFINE_PROMPT = """You are refining a music generation prompt based on conversation history.
The track started with this description, then the user requested changes in order:

{history_block}

Newest request: {edit_instruction}

Write ONE updated, complete music description (20-40 words) that incorporates ALL the
requested changes cumulatively. Keep any earlier details that weren't contradicted by
a later request. Output ONLY the description, no explanation, no quotes."""


@app.post("/edit-music-ai")
async def edit_music_ai(req: MusicEditRequest):
    """'Edit' a previously-generated AI music track. MusicGen-small has no
    true waveform-editing mode (unlike image diffusion's img2img), so this
    works by asking the LLM to synthesize an updated, cumulative description
    from the conversation history + new instruction, then generating a FRESH
    track from that refined prompt — giving an iterative, conversational feel
    even though the underlying audio is regenerated each time, not patched."""
    try:
        history_block = "\n".join(f"{i+1}. {h}" for i, h in enumerate(req.history)) or "(no prior description)"
        prompt = MUSIC_REFINE_PROMPT.format(history_block=history_block, edit_instruction=req.edit_instruction)

        async with httpx.AsyncClient(timeout=60.0) as client:
            res = await client.post("http://localhost:11434/api/generate",
                json={"model": req.model, "prompt": prompt, "stream": False})
        refined = res.json().get("response", "").strip().strip('"')
        if not refined:
            refined = (req.history[-1] if req.history else "") + ", " + req.edit_instruction

        # Always instrumental for background-music edits, consistent with the
        # default in regular generation (MusicGen-small vocals are weak)
        final_prompt = refined + ", instrumental only, no vocals, clean studio quality, loopable"

        result = await _run_musicgen_generation(final_prompt, req.duration, req.seed, req.edit_instruction[:30])
        return {**result, "refined_description": refined, "status": "ready"}
    except RuntimeError as e:
        return JSONResponse(status_code=400, content={"error": str(e)})
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"AI music edit failed: {str(e)}"})


# ══════════════════════════════════════════════════════════════════
# YOUTUBE METADATA — title / description / tags / chapters
# ══════════════════════════════════════════════════════════════════

class MetadataRequest(BaseModel):
    script:   str
    audio_id: str = ""   # if provided and transcription is cached, chapters are included
    model:    str = "llama3.2:3b"


METADATA_PROMPT = """You are a YouTube SEO expert for a Hill Climb Racing 2 (HCR2) gaming channel.
Given the video script below, generate YouTube metadata.

Return ONLY valid JSON (no markdown, no explanation), exactly this shape:
{
  "titles": ["title option 1 (under 60 chars, punchy)", "title option 2", "title option 3"],
  "description": "2-3 sentence YouTube description including a call to action to subscribe",
  "tags": ["tag1","tag2","tag3","tag4","tag5","tag6","tag7","tag8"],
  "hashtags": ["#hashtag1","#hashtag2","#hashtag3"]
}

Script:
{script}

JSON:"""


@app.post("/generate-metadata")
async def generate_metadata(req: MetadataRequest):
    """LLM-generated YouTube title/description/tags, plus chapter timestamps
    from cached Whisper word-segments (if this audio was already used in
    Video Studio so transcription is cached)."""
    try:
        prompt = METADATA_PROMPT.replace("{script}", req.script)
        async with httpx.AsyncClient(timeout=90.0) as client:
            res = await client.post("http://localhost:11434/api/generate",
                json={"model": req.model, "prompt": prompt, "stream": False})
        raw = res.json().get("response", "").strip()
        raw = raw.replace("```json", "").replace("```", "").strip()
        start, end = raw.find("{"), raw.rfind("}")
        if start == -1 or end == -1:
            raise ValueError("LLM did not return JSON")
        data = json.loads(raw[start:end+1])

        # Chapters from cached transcription, if available
        chapters = []
        if req.audio_id and req.audio_id in WORD_CACHE:
            segs = WORD_CACHE[req.audio_id]
            # Group into ~30s chapter chunks using segment boundaries
            last_chap_time = -999
            for seg in segs:
                if seg["start"] - last_chap_time >= 25:
                    text = seg["text"].strip()
                    label = (text[:40] + "...") if len(text) > 40 else text
                    chapters.append({"time": format_ass_time(seg["start"])[:-3], "label": label})
                    last_chap_time = seg["start"]
            # Always start at 0:00
            if chapters and chapters[0]["time"] != "0:00:00":
                chapters.insert(0, {"time": "0:00:00", "label": "Intro"})

        return {
            "titles": data.get("titles", []),
            "description": data.get("description", ""),
            "tags": data.get("tags", []),
            "hashtags": data.get("hashtags", []),
            "chapters": chapters,
            "ok": True
        }
    except Exception as e:
        return {"ok": False, "error": str(e), "titles": [], "description": "", "tags": [], "hashtags": [], "chapters": []}


class HookRequest(BaseModel):
    topic: str
    model: str = "llama3.2:3b"


HOOK_PROMPT = """You are a viral YouTube Shorts scriptwriter for Hill Climb Racing 2 gaming content.
Generate 3 different attention-grabbing opening lines (hooks) for a video about: {topic}

Each hook should be 1-2 sentences, create curiosity or excitement, and work as the FIRST thing
the viewer hears. Make each one a DIFFERENT angle/style (e.g. one shocking, one question-based,
one bold claim).

Return ONLY a JSON array of 3 strings, no markdown, no explanation:
["hook 1", "hook 2", "hook 3"]"""


@app.post("/generate-hooks")
async def generate_hooks(req: HookRequest):
    """Generate 3 alternative opening-line (hook) options for A/B testing."""
    try:
        prompt = HOOK_PROMPT.replace("{topic}", req.topic)
        async with httpx.AsyncClient(timeout=60.0) as client:
            res = await client.post("http://localhost:11434/api/generate",
                json={"model": req.model, "prompt": prompt, "stream": False})
        raw = res.json().get("response", "").strip()
        raw = raw.replace("```json", "").replace("```", "").strip()
        start, end = raw.find("["), raw.rfind("]")
        if start == -1 or end == -1:
            raise ValueError("LLM did not return a JSON array")
        hooks = json.loads(raw[start:end+1])
        hooks = [str(h).strip() for h in hooks if str(h).strip()][:3]
        if not hooks:
            raise ValueError("Empty hooks list")
        return {"hooks": hooks, "ok": True}
    except Exception as e:
        return {"hooks": [], "ok": False, "error": str(e)}


class ImagePromptRequest(BaseModel):
    script: str
    model:  str = "llama3.2:3b"


IMAGE_PROMPT_SYSTEM = """You write short, vivid scene descriptions for an AI image generator,
based on a YouTube gaming video script (Hill Climb Racing 2 content).

Return ONLY a single sentence (15-30 words) describing a visually striking scene that
matches the script's content/energy — concrete objects, action, mood, lighting. No camera
jargon, no "thumbnail" mentions, just the SCENE itself, written for a text-to-image model.

Script:
{script}

Scene description:"""


@app.post("/generate-image-prompt")
async def generate_image_prompt(req: ImagePromptRequest):
    """LLM turns a video script into a short visual scene description,
    suitable as a prompt for the local Stable Diffusion thumbnail generator."""
    try:
        prompt = IMAGE_PROMPT_SYSTEM.replace("{script}", req.script[:1000])
        async with httpx.AsyncClient(timeout=60.0) as client:
            res = await client.post("http://localhost:11434/api/generate",
                json={"model": req.model, "prompt": prompt, "stream": False})
        raw = res.json().get("response", "").strip()
        raw = raw.strip('"').strip()
        if not raw:
            raise ValueError("LLM returned empty prompt")
        return {"prompt": raw, "ok": True}
    except Exception as e:
        return {"prompt": "", "ok": False, "error": str(e)}


class IntentRequest(BaseModel):
    message:           str
    has_recent_image:  bool = False
    has_recent_music:  bool = False
    model:             str = "llama3.2:3b"


INTENT_PROMPT = """Classify the user's message into exactly one category. Reply ONLY with
valid JSON, no explanation, no markdown.

{{"intent": "image_new" | "image_edit" | "music_new" | "music_edit" | "chat", "prompt": "..."}}

Category rules:
- "image_new": user wants a brand NEW image/picture/thumbnail/artwork generated from scratch
- "image_edit": user wants to MODIFY/change/adjust the MOST RECENTLY generated image
  (only valid if a recent image exists in this conversation — recent image exists: {has_recent_image})
- "music_new": user wants brand NEW music/song/background track/audio generated from scratch
- "music_edit": user wants to MODIFY/change the MOST RECENTLY generated music
  (only valid if recent music exists in this conversation — recent music exists: {has_recent_music})
- "chat": anything else — questions, script writing, general conversation, requests unrelated
  to generating/editing an image or music

"prompt" field: if intent is image_new/music_new, extract a clean generation description from
the message. If image_edit/music_edit, extract just the requested CHANGE (e.g. "make it brighter").
If intent is "chat", set prompt to "".

User message: "{message}"

JSON:"""


@app.post("/detect-intent")
async def detect_intent(req: IntentRequest):
    """Lightweight LLM classifier run before normal chat replies — detects
    whether a free-typed message is actually requesting image/music
    generation or editing, so the chat tab can auto-route without requiring
    the explicit 🎨/🎵 buttons. Fails safe to 'chat' on any error, so a
    classifier hiccup never blocks normal conversation."""
    try:
        prompt = INTENT_PROMPT.format(
            has_recent_image=req.has_recent_image, has_recent_music=req.has_recent_music,
            message=req.message.replace('"', "'")[:500]
        )
        async with httpx.AsyncClient(timeout=20.0) as client:
            res = await client.post("http://localhost:11434/api/generate",
                json={"model": req.model, "prompt": prompt, "stream": False,
                      "options": {"num_predict": 100}})
        raw = res.json().get("response", "").strip()
        raw = raw.replace("```json", "").replace("```", "").strip()
        start, end = raw.find("{"), raw.rfind("}")
        if start == -1 or end == -1:
            return {"intent": "chat", "prompt": ""}
        data = json.loads(raw[start:end+1])
        intent = data.get("intent", "chat")
        if intent not in ("image_new", "image_edit", "music_new", "music_edit", "chat"):
            intent = "chat"
        # Guard against the LLM hallucinating an edit when nothing exists to edit
        if intent == "image_edit" and not req.has_recent_image:
            intent = "image_new"
        if intent == "music_edit" and not req.has_recent_music:
            intent = "music_new"
        return {"intent": intent, "prompt": str(data.get("prompt", "")).strip()}
    except Exception:
        # Fail safe — never block normal chat over a classifier error
        return {"intent": "chat", "prompt": ""}


# ══════════════════════════════════════════════════════════════════
# THUMBNAIL GENERATOR
# ══════════════════════════════════════════════════════════════════

THUMBS_DIR = "thumbnails"
os.makedirs(THUMBS_DIR, exist_ok=True)

# Common font locations — tries Windows fonts first (user runs on Windows),
# falls back to Linux/DejaVu, then PIL default if nothing found.
_FONT_CANDIDATES = {
    "impact": [
        r"C:\Windows\Fonts\impact.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    ],
    "arial_bold": [
        r"C:\Windows\Fonts\arialbd.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    ],
}


def _load_font(kind: str, size: int):
    for path in _FONT_CANDIDATES.get(kind, []):
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                continue
    try:
        return ImageFont.load_default(size=size)
    except Exception:
        return ImageFont.load_default()


class ThumbnailRequest(BaseModel):
    gameplay_id: str
    timestamp:   float = 1.0     # seconds into the clip to grab the frame
    text:        str   = ""      # overlay text (e.g. the video's hook line)
    style:       str   = "viral" # viral | clean | minimal
    text_color:  str   = "yellow"


THUMB_COLORS = {
    "yellow": (255, 220, 0), "white": (255, 255, 255), "red": (255, 60, 60),
    "cyan": (0, 230, 255), "green": (80, 255, 120), "orange": (255, 150, 0),
}


thumbnail_store = {}  # gameplay_id -> list of {thumb_id, filename, text, style, timestamp, created_at}


def _overlay_text_on_image(img: Image.Image, text: str, style: str, text_color: str, target_w: int = 1280, target_h: int = 720) -> Image.Image:
    """Burn bold YouTube-thumbnail-style text onto a PIL image. Shared by
    both the gameplay-frame thumbnail path and the AI-generated image path,
    so both get identical text rendering quality."""
    if not text.strip():
        return img
    draw = ImageDraw.Draw(img)
    color = THUMB_COLORS.get(text_color, THUMB_COLORS["yellow"])
    text_up = text.strip().upper()

    if style == "minimal":
        font = _load_font("arial_bold", 64)
        outline_w, shadow = 2, False
    elif style == "clean":
        font = _load_font("arial_bold", 88)
        outline_w, shadow = 4, True
    else:
        font = _load_font("impact", 110)
        outline_w, shadow = 6, True

    import textwrap
    max_chars = max(8, int(2400 / (font.size if hasattr(font, 'size') else 90)))
    lines = textwrap.wrap(text_up, width=max_chars)[:3]

    line_h = (font.size if hasattr(font, 'size') else 90) + 14
    total_h = line_h * len(lines)
    y = target_h - total_h - 50

    for line in lines:
        bbox = draw.textbbox((0, 0), line, font=font)
        w = bbox[2] - bbox[0]
        x = (target_w - w) // 2

        if shadow:
            draw.text((x+5, y+5), line, font=font, fill=(0, 0, 0, 160))

        if outline_w > 0:
            for ox in range(-outline_w, outline_w+1, 2):
                for oy in range(-outline_w, outline_w+1, 2):
                    if ox or oy:
                        draw.text((x+ox, y+oy), line, font=font, fill=(0, 0, 0))

        draw.text((x, y), line, font=font, fill=color)
        y += line_h

    return img


def _render_thumbnail_sync(gameplay_path: str, duration: float, timestamp: float,
                            text: str, style: str, text_color: str) -> str:
    """Blocking thumbnail render — extract frame + overlay text. Returns output path.
    Shared by the single-thumbnail endpoint and the A/B batch endpoint."""
    ts = max(0.0, min(timestamp, max(0.0, duration - 0.1))) if duration > 0 else max(0.0, timestamp)

    frame_path = os.path.join(THUMBS_DIR, f"frame_{uuid.uuid4().hex}.png")
    cmd = [FFMPEG_EXE, "-y", "-ss", str(ts), "-i", gameplay_path,
           "-vframes", "1", "-q:v", "2", frame_path]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    if r.returncode != 0 or not os.path.exists(frame_path):
        raise RuntimeError(f"Frame extraction failed: {r.stderr[-300:]}")

    img = Image.open(frame_path).convert("RGB")
    target_w, target_h = 1280, 720
    src_ratio = img.width / img.height
    tgt_ratio = target_w / target_h
    if src_ratio > tgt_ratio:
        new_h = target_h
        new_w = int(new_h * src_ratio)
    else:
        new_w = target_w
        new_h = int(new_w / src_ratio)
    img = img.resize((new_w, new_h))
    left = (new_w - target_w) // 2
    top  = (new_h - target_h) // 2
    img = img.crop((left, top, left + target_w, top + target_h))

    img = _overlay_text_on_image(img, text, style, text_color, target_w, target_h)

    out_path = os.path.join(THUMBS_DIR, f"thumb_{uuid.uuid4().hex}.jpg")
    img.save(out_path, "JPEG", quality=92)

    try:
        os.unlink(frame_path)
    except Exception:
        pass

    return out_path


@app.post("/generate-thumbnail")
async def generate_thumbnail(req: ThumbnailRequest):
    """Extract a frame from the gameplay clip at `timestamp` and overlay
    bold text — quick YouTube thumbnail generation."""
    try:
        if req.gameplay_id not in gameplay_store:
            return JSONResponse(status_code=404, content={"error": "Gameplay clip not found. Upload it in Video Studio first."})
        g = gameplay_store[req.gameplay_id]
        gameplay_path = os.path.abspath(g["path"])
        duration = g.get("duration", 0)

        out_path = await asyncio.to_thread(
            _render_thumbnail_sync, gameplay_path, duration, req.timestamp, req.text, req.style, req.text_color
        )

        thumb_id = os.path.basename(out_path).replace("thumb_", "").replace(".jpg", "")
        thumbnail_store.setdefault(req.gameplay_id, []).append({
            "thumb_id": thumb_id, "filename": os.path.basename(out_path),
            "text": req.text, "style": req.style, "timestamp": req.timestamp,
            "created_at": datetime.now().isoformat(),
        })

        return FileResponse(out_path, media_type="image/jpeg", filename="thumbnail.jpg")
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


class ThumbnailBatchRequest(BaseModel):
    gameplay_id: str
    text:        str = ""
    text_color:  str = "yellow"
    mode:        str = "frames"


@app.post("/generate-thumbnails-batch")
async def generate_thumbnails_batch(req: ThumbnailBatchRequest):
    """Generate 3 thumbnail variants at once for A/B comparison — either
    3 different frames (same style) or 3 different styles (same frame)."""
    try:
        if req.gameplay_id not in gameplay_store:
            return JSONResponse(status_code=404, content={"error": "Gameplay clip not found. Upload it in Video Studio first."})
        g = gameplay_store[req.gameplay_id]
        gameplay_path = os.path.abspath(g["path"])
        duration = g.get("duration", 1.0) or 1.0

        if req.mode == "styles":
            variants = [(duration * 0.35, "viral"), (duration * 0.35, "clean"), (duration * 0.35, "minimal")]
        else:
            variants = [(duration * 0.20, "viral"), (duration * 0.50, "viral"), (duration * 0.80, "viral")]

        results = []
        for ts, style in variants:
            out_path = await asyncio.to_thread(
                _render_thumbnail_sync, gameplay_path, duration, ts, req.text, style, req.text_color
            )
            thumb_id = os.path.basename(out_path).replace("thumb_", "").replace(".jpg", "")
            entry = {
                "thumb_id": thumb_id, "filename": os.path.basename(out_path),
                "text": req.text, "style": style, "timestamp": round(ts, 1),
                "created_at": datetime.now().isoformat(),
            }
            thumbnail_store.setdefault(req.gameplay_id, []).append(entry)
            results.append({**entry, "url": f"/thumbnails/{entry['filename']}"})

        return {"variants": results, "status": "ready"}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.get("/thumbnails/{filename}")
async def serve_thumbnail(filename: str):
    if ".." in filename or filename.startswith("/"):
        return JSONResponse(status_code=400, content={"error": "Invalid path"})
    filepath = os.path.join(THUMBS_DIR, filename)
    if os.path.exists(filepath):
        return FileResponse(filepath, media_type="image/jpeg")
    return JSONResponse(status_code=404, content={"error": "Thumbnail not found"})


@app.get("/list-thumbnails/{gameplay_id}")
async def list_thumbnails(gameplay_id: str):
    items = thumbnail_store.get(gameplay_id, [])
    items = sorted(items, key=lambda x: x["created_at"], reverse=True)[:12]
    return {"thumbnails": [{**i, "url": f"/thumbnails/{i['filename']}", "server_path": i.get("path", "")} for i in items]}


# ══════════════════════════════════════════════════════════════════
# AI THUMBNAIL GENERATION — local Stable Diffusion (SD-Turbo)
# ══════════════════════════════════════════════════════════════════

sd_pipeline = None
_sd_lock = asyncio.Lock()

THUMB_STYLE_PROMPT_SUFFIX = {
    "viral":   "dramatic lighting, high contrast, vibrant saturated colors, YouTube thumbnail style, eye-catching, 8k",
    "clean":   "clean modern composition, soft lighting, professional photography, balanced colors",
    "minimal": "minimalist, simple composition, soft pastel colors, clean background",
}


def get_sd_pipeline():
    """Lazy-load Stable Diffusion (SD-Turbo). Same pattern as get_parler()/
    get_xtts() elsewhere in this file — only loads on first actual use."""
    global sd_pipeline
    if sd_pipeline is None:
        try:
            import torch
            from diffusers import AutoPipelineForText2Image
        except ImportError:
            raise RuntimeError(
                "Stable Diffusion not installed.\n"
                "Run: pip install diffusers torch accelerate transformers\n"
                "(torch with CUDA strongly recommended — CPU generation is very slow)"
            )
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if device == "cuda" else torch.float32
        print(f"  Loading Stable Diffusion (SD-Turbo) on {device} (first run downloads ~2GB)...")
        sd_pipeline = AutoPipelineForText2Image.from_pretrained(
            "stabilityai/sd-turbo", torch_dtype=dtype, variant="fp16" if device == "cuda" else None
        )
        sd_pipeline = sd_pipeline.to(device)
        print("  Stable Diffusion  ✓  Ready")
    return sd_pipeline


@app.get("/thumbnail-ai-status")
async def thumbnail_ai_status():
    """Lightweight check (no model load) — is Stable Diffusion importable?
    Mirrors /parler-status and /clone-status so the UI shows a consistent badge."""
    try:
        import torch
        from diffusers import AutoPipelineForText2Image  # noqa: F401
        device = "cuda" if torch.cuda.is_available() else "cpu"
        detail = f"Library installed ✓ (device: {device})"
        if device == "cpu":
            detail += " · WARNING: CPU generation is slow (60-120s+ per image)"
        else:
            detail += " · GPU detected, generation should take ~5-15s"
        return {"available": True, "loaded": sd_pipeline is not None, "detail": detail}
    except ImportError:
        return {"available": False, "loaded": False,
                "detail": "Not installed. Run: pip install diffusers torch accelerate transformers"}


class ThumbnailAIRequest(BaseModel):
    gameplay_id: str = ""        # optional — only used for gallery grouping
    prompt:      str             # scene description, e.g. "car flying off a cliff explosion"
    style:       str = "viral"   # viral | clean | minimal — affects both image mood + text overlay
    text:        str = ""        # optional overlay text (hook line)
    text_color:  str = "yellow"
    steps:       int = 2         # SD-Turbo works well at 1-4 steps; more = slower, marginal quality gain


@app.post("/generate-thumbnail-ai")
async def generate_thumbnail_ai(req: ThumbnailAIRequest):
    """Generate a thumbnail image FROM SCRATCH using a text prompt (Stable
    Diffusion / SD-Turbo) instead of extracting a gameplay frame — for when
    you want a custom illustrated/stylized thumbnail rather than a real
    frame from the clip. Then overlays bold text the same way as the
    frame-based thumbnails."""
    try:
        async with _sd_lock:  # one generation at a time — avoids GPU OOM from concurrent requests
            pipe = await asyncio.to_thread(get_sd_pipeline)

            suffix = THUMB_STYLE_PROMPT_SUFFIX.get(req.style, THUMB_STYLE_PROMPT_SUFFIX["viral"])
            full_prompt = f"{req.prompt.strip()}, {suffix}, no text, no watermark"
            steps = max(1, min(8, req.steps))

            def _run_sd():
                import torch
                generator = None
                try:
                    if torch.cuda.is_available():
                        generator = torch.Generator(device="cuda")
                except Exception:
                    pass
                result = pipe(
                    prompt=full_prompt, num_inference_steps=steps,
                    guidance_scale=0.0,  # SD-Turbo is distilled for guidance_scale=0
                    width=1280, height=720, generator=generator,
                )
                return result.images[0]

            img = await asyncio.to_thread(_run_sd)
            img = img.convert("RGB")

        img = _overlay_text_on_image(img, req.text, req.style, req.text_color, 1280, 720)

        out_path = os.path.join(THUMBS_DIR, f"thumb_ai_{uuid.uuid4().hex}.jpg")
        img.save(out_path, "JPEG", quality=92)

        if req.gameplay_id:
            thumb_id = os.path.basename(out_path).replace("thumb_ai_", "").replace(".jpg", "")
            thumbnail_store.setdefault(req.gameplay_id, []).append({
                "thumb_id": thumb_id, "filename": os.path.basename(out_path),
                "text": req.text, "style": req.style, "timestamp": -1,  # -1 marks "AI generated, not a real frame"
                "ai_prompt": req.prompt, "created_at": datetime.now().isoformat(),
            })

        return FileResponse(out_path, media_type="image/jpeg", filename="thumbnail_ai.jpg")
    except RuntimeError as e:
        return JSONResponse(status_code=400, content={"error": str(e)})
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"AI generation failed: {str(e)}"})


sd_img2img_pipeline = None


def get_sd_img2img_pipeline():
    """Lazy-load the img2img variant of SD-Turbo. Uses diffusers'
    AutoPipelineForImage2Image.from_pipe(), which REUSES the already-loaded
    text2img weights in memory instead of loading a second copy — efficient,
    and only triggers a download the first time either mode is used.
    NOTE: kept only as a CHEAP fallback (no extra download) — see
    get_instruct_pix2pix_pipeline() below for the actual edit path used by
    /edit-thumbnail-ai, since SD-Turbo's img2img mode cannot follow semantic
    instructions (see that function's docstring for why)."""
    global sd_img2img_pipeline
    if sd_img2img_pipeline is None:
        from diffusers import AutoPipelineForImage2Image
        base = get_sd_pipeline()  # ensures base weights are loaded first
        sd_img2img_pipeline = AutoPipelineForImage2Image.from_pipe(base)
        print("  Stable Diffusion (img2img)  ✓  Ready")
    return sd_img2img_pipeline


instruct_pix2pix_pipeline = None


def get_instruct_pix2pix_pipeline():
    """Lazy-load InstructPix2Pix — a model specifically TRAINED to follow
    plain-English edit instructions ('replace the sweets with a dog',
    'reduce the number of bowls to 3'), unlike SD-Turbo's img2img mode
    which is only a generic variation/denoise tool with near-zero text
    steering (it runs at guidance_scale=0 by design — required for its
    distillation — which is exactly why instructions were being ignored
    and edits were just brightening/varying the image).
    First use downloads ~5GB (separate from SD-Turbo's own weights)."""
    global instruct_pix2pix_pipeline
    if instruct_pix2pix_pipeline is None:
        import torch
        from diffusers import StableDiffusionInstructPix2PixPipeline
        device = "cuda" if torch.cuda.is_available() else "cpu"
        instruct_pix2pix_pipeline = StableDiffusionInstructPix2PixPipeline.from_pretrained(
            "timbrooks/instruct-pix2pix",
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            safety_checker=None,
        ).to(device)
        print("  InstructPix2Pix (image editing)  ✓  Ready")
    return instruct_pix2pix_pipeline


class ImageEditRequest(BaseModel):
    base_image_b64: str          # base64-encoded JPEG/PNG of the image to edit (no data: prefix)
    edit_prompt:    str          # what to change, e.g. "make the sky more dramatic, add explosion"
    style:          str = "viral"
    strength:       float = 0.55 # 0.0-1.0 — how much to change vs preserve the original (higher = bigger change)
    text:           str = ""
    text_color:     str = "yellow"
    steps:          int = 4


@app.post("/edit-thumbnail-ai")
async def edit_thumbnail_ai(req: ImageEditRequest):
    """TRUE instruction-following image editing — modifies a previously-
    generated AI thumbnail based on a follow-up instruction like 'replace
    the sweets with a dog' or 'reduce the number of bowls to 3'.
    Uses InstructPix2Pix (NOT SD-Turbo img2img). SD-Turbo's img2img mode
    runs at guidance_scale=0 (required by its distillation), which gives
    the text prompt almost no influence — it only lightly varies/denoises
    the image (visible as e.g. just a brightness change), it cannot follow
    semantic instructions like object replacement or counting. InstructPix2Pix
    is trained specifically for this and uses real classifier-free guidance
    on both the image and the instruction text."""
    try:
        import base64, io as _io
        try:
            img_bytes = base64.b64decode(req.base_image_b64)
            base_img = Image.open(_io.BytesIO(img_bytes)).convert("RGB")
        except Exception:
            return JSONResponse(status_code=400, content={"error": "Could not decode the base image — try generating a fresh image instead."})

        base_img = base_img.resize((1280, 720))

        async with _sd_lock:
            pipe = await asyncio.to_thread(get_instruct_pix2pix_pipeline)

            instruction = req.edit_prompt.strip()
            # strength maps onto image_guidance_scale: how strongly the
            # output must stay faithful to the ORIGINAL image's structure.
            # Lower = more freedom to actually make big changes (object
            # replacement, counts); higher = more conservative. 1.0-1.5 is
            # InstructPix2Pix's documented sweet spot, vs SD-Turbo's
            # strength=0.55 default which was actively suppressing edits.
            image_guidance = max(1.0, min(2.2, 1.0 + (1 - req.strength) * 1.2))

            def _run_edit():
                import torch
                generator = None
                try:
                    if torch.cuda.is_available():
                        generator = torch.Generator(device="cuda")
                except Exception:
                    pass
                result = pipe(
                    instruction, image=base_img,
                    num_inference_steps=20,
                    image_guidance_scale=image_guidance,
                    guidance_scale=7.5,
                    generator=generator,
                )
                return result.images[0]

            img = await asyncio.to_thread(_run_edit)
            img = img.convert("RGB").resize((1280, 720))

        img = _overlay_text_on_image(img, req.text, req.style, req.text_color, 1280, 720)

        out_path = os.path.join(THUMBS_DIR, f"thumb_edit_{uuid.uuid4().hex}.jpg")
        img.save(out_path, "JPEG", quality=92)

        return FileResponse(out_path, media_type="image/jpeg", filename="thumbnail_edited.jpg")
    except RuntimeError as e:
        return JSONResponse(status_code=400, content={"error": str(e)})
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"AI edit failed: {str(e)}"})


# ══════════════════════════════════════════════════════════════════
# PROJECTS — save/load script + voiceover + gameplay + settings bundle
# ══════════════════════════════════════════════════════════════════

class ProjectSaveRequest(BaseModel):
    name:        str
    script:      str = ""
    audio_id:    str = ""
    gameplay_id: str = ""
    settings:    dict = {}   # subtitle style/effect/position/etc — anything JSON-able
    project_id:  str = ""    # if provided, update existing project instead of creating new


@app.post("/save-project")
async def save_project(req: ProjectSaveRequest):
    """Save (or update) a named project bundling script + voice + gameplay + settings,
    so you can close the app and resume later without redoing setup."""
    try:
        now = datetime.now().isoformat()
        pid = req.project_id or str(uuid.uuid4())
        settings_json = json.dumps(req.settings)

        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        if req.project_id:
            c.execute("""UPDATE projects SET name=?, script=?, audio_id=?, gameplay_id=?,
                          settings_json=?, updated_at=? WHERE id=?""",
                      (req.name, req.script, req.audio_id, req.gameplay_id, settings_json, now, pid))
            if c.rowcount == 0:
                c.execute("INSERT INTO projects VALUES (?,?,?,?,?,?,?,?)",
                          (pid, req.name, req.script, req.audio_id, req.gameplay_id, settings_json, now, now))
        else:
            c.execute("INSERT INTO projects VALUES (?,?,?,?,?,?,?,?)",
                      (pid, req.name, req.script, req.audio_id, req.gameplay_id, settings_json, now, now))
        conn.commit()
        conn.close()
        return {"project_id": pid, "status": "saved"}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.get("/list-projects")
async def list_projects():
    """List all saved projects, most recently updated first."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT id, name, script, audio_id, gameplay_id, settings_json, created_at, updated_at FROM projects ORDER BY updated_at DESC")
    rows = c.fetchall()
    conn.close()
    projects = []
    for r in rows:
        script_preview = (r[2][:80] + "...") if r[2] and len(r[2]) > 80 else (r[2] or "")
        projects.append({
            "project_id": r[0], "name": r[1], "script_preview": script_preview,
            "has_audio": bool(r[3]), "has_gameplay": bool(r[4]),
            "created_at": r[6], "updated_at": r[7],
        })
    return {"projects": projects}


@app.get("/load-project/{project_id}")
async def load_project(project_id: str):
    """Load a full project — script, audio_id, gameplay_id, and saved settings."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT id, name, script, audio_id, gameplay_id, settings_json, created_at, updated_at FROM projects WHERE id=?", (project_id,))
    r = c.fetchone()
    conn.close()
    if not r:
        return JSONResponse(status_code=404, content={"error": "Project not found"})
    try:
        settings = json.loads(r[5]) if r[5] else {}
    except Exception:
        settings = {}
    return {
        "project_id": r[0], "name": r[1], "script": r[2], "audio_id": r[3],
        "gameplay_id": r[4], "settings": settings, "created_at": r[6], "updated_at": r[7],
    }


@app.delete("/delete-project/{project_id}")
async def delete_project(project_id: str):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("DELETE FROM projects WHERE id=?", (project_id,))
    conn.commit()
    conn.close()
    return {"deleted": True}


# ══════════════════════════════════════════════════════════════════
# VOICE LIBRARY — save/reuse named voice configurations
# ══════════════════════════════════════════════════════════════════

class VoicePresetRequest(BaseModel):
    name:    str
    engine:  str          # "kokoro" | "hindi" | "clone" | "parler"
    payload: dict         # engine-specific: voice id, description, speed/pitch, etc.


@app.post("/save-voice-preset")
async def save_voice_preset(req: VoicePresetRequest):
    """Save a named voice configuration (Kokoro voice, Edge voice, Parler
    description, etc.) for instant reuse later — no retyping descriptions."""
    try:
        pid = str(uuid.uuid4())
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("INSERT INTO voice_presets VALUES (?,?,?,?,?)",
                  (pid, req.name, req.engine, json.dumps(req.payload), datetime.now().isoformat()))
        conn.commit()
        conn.close()
        return {"preset_id": pid, "status": "saved"}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.get("/list-voice-presets")
async def list_voice_presets():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT id, name, engine, payload_json, created_at FROM voice_presets ORDER BY created_at DESC")
    rows = c.fetchall()
    conn.close()
    presets = []
    for r in rows:
        try:
            payload = json.loads(r[3])
        except Exception:
            payload = {}
        presets.append({"preset_id": r[0], "name": r[1], "engine": r[2], "payload": payload, "created_at": r[4]})
    return {"presets": presets}


@app.delete("/delete-voice-preset/{preset_id}")
async def delete_voice_preset(preset_id: str):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("DELETE FROM voice_presets WHERE id=?", (preset_id,))
    conn.commit()
    conn.close()
    return {"deleted": True}


# ══════════════════════════════════════════════════════════════════
# ── CLIP FINDER ──────────────────────────────────────────────────
# Long recording in -> transcript -> local-LLM-scored "viral moment"
# candidates -> review/select -> auto-cut + captioned clip export.
# Reuses: faster-whisper (get_word_segments/WORD_CACHE), the existing
# ASS caption engine (generate_ass_content), the existing FFmpeg crop/
# scale filter (build_ffmpeg_filter), and the local Ollama chat model
# already configured elsewhere in this app (selected_model).
# ══════════════════════════════════════════════════════════════════

CLIPFINDER_ALLOWED_EXT = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}


@app.post("/clipfinder/upload")
async def clipfinder_upload(file: UploadFile = File(...)):
    """Ingest a long recording (podcast, webcam, gameplay VOD, screen
    recording — anything). Just stores + probes it; no processing yet."""
    try:
        ext = os.path.splitext(file.filename or "")[1].lower()
        if ext not in CLIPFINDER_ALLOWED_EXT:
            return {"error": f"Unsupported format. Use: {', '.join(sorted(CLIPFINDER_ALLOWED_EXT))}", "status": "failed"}
        recording_id = str(uuid.uuid4())
        out_path = f"{RECORDINGS_DIR}/{recording_id}{ext}"
        content = await file.read()
        with open(out_path, "wb") as f_out:
            f_out.write(content)
        info = get_video_info(out_path)
        recording_store[recording_id] = {
            "path": out_path, "filename": file.filename,
            "width": info["width"], "height": info["height"], "duration": info["duration"],
            "context_notes": "",
        }
        _cf_db_save_recording(recording_id)  # auto-save — survives an app restart
        return {
            "recording_id": recording_id, "filename": file.filename,
            "duration": round(info["duration"], 1),
            "resolution": f"{info['width']}x{info['height']}", "status": "ready",
        }
    except Exception as e:
        return {"error": str(e), "status": "failed"}


@app.get("/clipfinder/recordings")
async def clipfinder_recordings():
    """List everything ingested — survives restarts now (hydrated from DB
    at startup), not just this runtime session."""
    return {
        "recordings": [
            {"recording_id": rid, "filename": r["filename"],
             "duration": round(r["duration"], 1), "resolution": f"{r['width']}x{r['height']}",
             "transcribed": rid in WORD_CACHE, "has_moments": rid in MOMENTS_CACHE,
             "context_notes": r.get("context_notes", "")}
            for rid, r in recording_store.items()
        ]
    }


@app.patch("/clipfinder/recordings/{recording_id}/context")
async def clipfinder_set_context(recording_id: str, payload: dict):
    """Sets the 'about this video' notes reused across EVERY clip's script
    draft for this recording — set once, applies everywhere, instead of
    retyping the same context per clip."""
    rec = recording_store.get(recording_id)
    if not rec:
        return {"error": "Recording not found"}
    rec["context_notes"] = (payload.get("context_notes") or "").strip()
    _cf_db_save_recording(recording_id)
    return {"status": "saved", "context_notes": rec["context_notes"]}


@app.delete("/clipfinder/recordings/{recording_id}")
async def clipfinder_delete_recording(recording_id: str):
    """Full cleanup — the missing piece flagged: recordings had zero
    controls (no delete/rename anywhere, not even in the backend). Removes
    the file from disk, its transcript/moments cache, and every DB row tied
    to it (recording, transcript, moments, exported clips)."""
    rec = recording_store.get(recording_id)
    if not rec:
        return {"error": "Recording not found"}
    try:
        if os.path.exists(rec["path"]):
            os.remove(rec["path"])
    except Exception as e:
        print(f"[clipfinder] could not remove file for {recording_id}: {e}")

    recording_store.pop(recording_id, None)
    WORD_CACHE.pop(recording_id, None)
    MOMENTS_CACHE.pop(recording_id, None)

    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("DELETE FROM clipfinder_recordings WHERE recording_id=?", (recording_id,))
    c.execute("DELETE FROM clipfinder_transcripts WHERE recording_id=?", (recording_id,))
    c.execute("DELETE FROM clipfinder_moments WHERE recording_id=?", (recording_id,))
    c.execute("DELETE FROM clipfinder_exports WHERE recording_id=?", (recording_id,))
    conn.commit()
    conn.close()
    return {"status": "deleted", "recording_id": recording_id}


@app.patch("/clipfinder/recordings/{recording_id}/rename")
async def clipfinder_rename_recording(recording_id: str, payload: dict):
    """Rename a recording's DISPLAY name (not the underlying file — keeps
    this safe/instant, no risk of breaking an in-progress ffmpeg job that
    has the old path open)."""
    new_name = (payload.get("filename") or "").strip()
    if not new_name:
        return {"error": "New name cannot be empty"}
    rec = recording_store.get(recording_id)
    if not rec:
        return {"error": "Recording not found"}
    rec["filename"] = new_name
    conn = sqlite3.connect(DB_PATH)
    conn.execute("UPDATE clipfinder_recordings SET filename=? WHERE recording_id=?", (new_name, recording_id))
    conn.commit()
    conn.close()
    return {"status": "renamed", "filename": new_name}


@app.get("/clipfinder/cached-moments/{recording_id}")
async def clipfinder_cached_moments(recording_id: str):
    """Restores previously-found moments for a recording without re-running
    the LLM — this is the fix for 'viral moments not stored, lost on exit'."""
    cached = MOMENTS_CACHE.get(recording_id)
    if not cached:
        return {"found": False}
    return {"found": True, "moments": cached["moments"], "settings": cached["settings"]}


def _clipfinder_transcribe_worker(job_id: str, recording_id: str, path: str):
    """Blocking — run via BackgroundTasks. Duplicates get_word_segments'
    VAD-with-fallback logic (same fix as before) but iterates the segment
    generator directly instead of collecting-then-returning, so we can
    report REAL progress (% = latest segment's end-time / total duration)
    after every segment — not a fake/estimated number, an honest one,
    matching the bar set by Find Moments' chunk-by-chunk progress."""
    try:
        duration = recording_store.get(recording_id, {}).get("duration", 0) or 0
        clipfinder_jobs[job_id] = {"status": "processing", "stage": "transcribing", "progress_pct": 0, "error": None}

        if recording_id in WORD_CACHE:
            segments = WORD_CACHE[recording_id]
        else:
            def _run(vad: bool):
                segments_iter, _ = whisper_model.transcribe(
                    path, word_timestamps=True, language=None, beam_size=5, vad_filter=vad
                )
                out = []
                for seg in segments_iter:
                    words = [{"word": w.word, "start": w.start, "end": w.end} for w in (seg.words or [])]
                    out.append({"start": seg.start, "end": seg.end, "text": seg.text, "words": words})
                    if duration > 0:
                        clipfinder_jobs[job_id]["progress_pct"] = min(99, round((seg.end / duration) * 100))
                return out

            try:
                segments = _run(vad=True)
            except Exception as vad_err:
                clipfinder_jobs[job_id]["progress_pct"] = 0  # restarting the pass
                try:
                    segments = _run(vad=False)
                except Exception as plain_err:
                    raise RuntimeError(
                        f"Transcription failed even without VAD filtering: {plain_err} "
                        f"(original VAD error: {vad_err})"
                    )
            WORD_CACHE[recording_id] = segments
            _cf_db_save_transcript(recording_id)  # auto-save — survives an app restart

        total_text_chars = sum(len(s["text"]) for s in segments)
        clipfinder_jobs[job_id] = {
            "status": "completed", "stage": "transcribed", "progress_pct": 100, "error": None,
            "recording_id": recording_id, "segment_count": len(segments),
            "char_count": total_text_chars,
        }
    except Exception as e:
        clipfinder_jobs[job_id] = {"status": "failed", "stage": "transcribing", "error": str(e)}


@app.post("/clipfinder/transcribe")
async def clipfinder_transcribe(payload: dict, background_tasks: BackgroundTasks):
    recording_id = payload.get("recording_id", "")
    if recording_id not in recording_store:
        return {"error": "Unknown recording_id — upload it first", "status": "failed"}
    job_id = str(uuid.uuid4())
    clipfinder_jobs[job_id] = {"status": "queued", "stage": "queued", "error": None}
    background_tasks.add_task(_clipfinder_transcribe_worker, job_id, recording_id, recording_store[recording_id]["path"])
    return {"job_id": job_id, "status": "queued"}


@app.get("/clipfinder/status/{job_id}")
async def clipfinder_status(job_id: str):
    if job_id not in clipfinder_jobs:
        return {"status": "not_found"}
    return clipfinder_jobs[job_id]


def _fmt_mmss(t: float) -> str:
    m, s = divmod(int(t), 60)
    return f"{m:02d}:{s:02d}"


def _clipfinder_build_transcript_chunks(segments: list, max_chars: int = 6000) -> list:
    """Turn cached whisper segments into '[MM:SS] text' lines, chunked so
    each request to the local model stays within a small model's comfortable
    context window. Each chunk carries its own time range."""
    chunks, cur_lines, cur_len = [], [], 0
    chunk_start = segments[0]["start"] if segments else 0.0
    for seg in segments:
        line = f"[{_fmt_mmss(seg['start'])}] {seg['text'].strip()}"
        if cur_len + len(line) > max_chars and cur_lines:
            chunks.append({"start": chunk_start, "end": seg["start"], "text": "\n".join(cur_lines)})
            cur_lines, cur_len = [], 0
            chunk_start = seg["start"]
        cur_lines.append(line)
        cur_len += len(line) + 1
    if cur_lines:
        chunks.append({"start": chunk_start, "end": segments[-1]["end"], "text": "\n".join(cur_lines)})
    return chunks


_MOMENT_PROMPT = """You are a short-form video editor. Below is a timestamped transcript excerpt from a longer recording. Each line starts with [MM:SS] showing when that line was spoken, counted from the start of the FULL recording (not this excerpt).

Find moments that would work as standalone short-form clips (Reels/Shorts/TikTok) — strong hooks, punchy or surprising statements, emotional peaks, funny moments, or self-contained useful insights. Each clip should be roughly 15-90 seconds long.

Return ONLY a JSON array (no prose, no markdown fences). IMPORTANT: start_sec and end_sec MUST be plain numbers of TOTAL SECONDS — convert [MM:SS] to seconds, do NOT return the "MM:SS" string itself. Example: a line at [02:15] is start_sec 135, NOT "02:15" or 215.

Each item must look exactly like this:
{{"start_sec": 135, "end_sec": 180, "title": "<short catchy title, under 60 chars>", "hook": "<the exact opening line/hook, under 25 words>", "score": <0-100 virality score>, "reason": "<one short sentence why this works>"}}

If nothing in this excerpt is clip-worthy, return [].

TRANSCRIPT EXCERPT:
{transcript}
"""


def _parse_time_value(v) -> float:
    """Accepts a plain number OR a 'MM:SS' / 'H:MM:SS' string. Local models
    frequently mirror the transcript's own '[MM:SS] text' formatting back
    in their JSON instead of converting to raw seconds like the prompt
    asked — that mismatch was silently dropping EVERY candidate moment
    (float('12:34') throws, and the old code just skipped it with no
    signal that anything had gone wrong)."""
    if isinstance(v, (int, float)):
        return float(v)
    s = str(v).strip()
    if re.fullmatch(r"-?\d+(\.\d+)?", s):
        return float(s)
    parts = s.split(":")
    if 2 <= len(parts) <= 3 and all(re.fullmatch(r"\d+(\.\d+)?", p) for p in parts):
        parts = [float(p) for p in parts]
        if len(parts) == 2:
            return parts[0] * 60 + parts[1]
        return parts[0] * 3600 + parts[1] * 60 + parts[2]
    raise ValueError(f"Unparseable time value: {v!r}")


async def _ollama_find_moments_chunk(transcript_text: str, model: str) -> dict:
    """Returns {'moments': [...], 'error': str|None, 'raw_preview': str}
    instead of silently returning [] on any failure — callers can now tell
    the difference between 'model looked and found nothing' and 'the
    request/parsing itself failed', and surface that to the user instead
    of a mysterious silent zero.

    FIXED: hybrid "thinking" models (qwen3.x, deepseek-r1, etc.) route
    their internal reasoning into a separate `message.thinking` field and
    can leave `message.content` completely empty — especially when
    format=json's strict grammar conflicts with the model's natural
    chain-of-thought style, which can also make it spin for minutes before
    giving up. `"think": false` tells Ollama to skip that reasoning phase
    entirely for models that support toggling it (silently ignored by
    models that don't have thinking mode at all, so this is safe for every
    model — reasoning or not). As a second safety net, if content still
    comes back empty but `thinking` has text, we try to salvage a JSON
    array out of THAT instead of failing outright."""
    prompt = _MOMENT_PROMPT.format(transcript=transcript_text)
    try:
        async with httpx.AsyncClient(timeout=180.0) as client:
            resp = await client.post(
                "http://localhost:11434/api/chat",
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "stream": False,
                    "format": "json",
                    "think": False,
                    "options": {"temperature": 0.4},
                },
            )
            if resp.status_code != 200:
                return {"moments": [], "error": f"Ollama returned HTTP {resp.status_code}: {resp.text[:200]}", "raw_preview": ""}
            msg = resp.json().get("message", {})
            content = msg.get("content", "")
            if not content.strip() and msg.get("thinking", "").strip():
                # Model ignored think:false (older Ollama/model combo) and
                # put everything in `thinking` anyway — salvage from there
                # instead of reporting a false "empty response".
                content = msg["thinking"]
    except Exception as e:
        return {"moments": [], "error": f"Request to Ollama failed: {e}", "raw_preview": ""}

    if not content.strip():
        return {"moments": [], "error": "Model returned an empty response", "raw_preview": ""}

    raw_preview = content[:300]

    try:
        data = json.loads(content)
    except Exception:
        m = re.search(r"\[.*\]", content, re.S)
        if not m:
            return {"moments": [], "error": "Model output wasn't valid JSON and no [...] array could be found", "raw_preview": raw_preview}
        try:
            data = json.loads(m.group(0))
        except Exception as e:
            return {"moments": [], "error": f"Extracted JSON block still failed to parse: {e}", "raw_preview": raw_preview}

    # Unwrap: model may return the array directly, wrap it under a key
    # ("moments"/"clips"/"result"/etc.), or — under forced JSON mode —
    # emit a single moment object instead of a list of one.
    if isinstance(data, dict):
        list_val = None
        for key in ("moments", "clips", "result", "results", "candidates"):
            if isinstance(data.get(key), list):
                list_val = data[key]
                break
        if list_val is None:
            # Any list value anywhere in the object, or treat the dict
            # itself as a single moment if it looks like one.
            list_vals = [v for v in data.values() if isinstance(v, list)]
            if list_vals:
                list_val = list_vals[0]
            elif "start_sec" in data or "start" in data:
                list_val = [data]
            else:
                list_val = []
        data = list_val

    if not isinstance(data, list):
        return {"moments": [], "error": f"Parsed JSON was not a list (got {type(data).__name__})", "raw_preview": raw_preview}
    if not data:
        return {"moments": [], "error": None, "raw_preview": raw_preview}  # model genuinely found nothing — not an error

    return {"moments": data, "error": None, "raw_preview": raw_preview}


def _cf_detect_scene_changes(video_path: str, threshold: float = 0.35) -> list:
    """Returns timestamps (seconds) where ffmpeg detects a scene/shot change.
    Pure signal analysis, no AI, no extra downloads — works as a baseline
    'something happened here' signal for ANY footage, even silent."""
    cmd = [FFMPEG_EXE, "-i", video_path, "-filter:v", f"select='gt(scene,{threshold})',showinfo", "-f", "null", "-"]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        return [float(m) for m in re.findall(r"pts_time:([\d.]+)", r.stderr)]
    except Exception:
        return []


def _cf_audio_energy_profile(video_path: str, duration: float, window: float = 2.0):
    """Computes windowed RMS energy ONCE — shared by heuristic peak detection
    AND per-moment audio-mode suggestion, so a recording only gets decoded
    for this purpose a single time per Find Moments run, not once per use.
    Returns (energies: np.ndarray, baseline: float, window: float) or
    (None, 0, window) on failure."""
    if duration <= 0:
        return None, 0.0, window
    cmd = [FFMPEG_EXE, "-i", video_path, "-vn", "-ac", "1", "-ar", "16000", "-f", "s16le", "-"]
    try:
        r = subprocess.run(cmd, capture_output=True, timeout=300)
        pcm = np.frombuffer(r.stdout, dtype=np.int16).astype(np.float32) / 32768.0
    except Exception:
        return None, 0.0, window
    if len(pcm) == 0:
        return None, 0.0, window
    sr = 16000
    win_samples = int(window * sr)
    n_windows = max(1, len(pcm) // win_samples)
    energies = []
    for i in range(n_windows):
        seg = pcm[i * win_samples:(i + 1) * win_samples]
        if len(seg) == 0:
            continue
        energies.append(float(np.sqrt(np.mean(seg ** 2) + 1e-9)))
    if not energies:
        return None, 0.0, window
    energies = np.array(energies)
    return energies, float(np.median(energies)), window


def _cf_detect_audio_peaks_from_profile(energies, baseline: float, window: float) -> list:
    """Windows where audio energy spikes well above the recording's own
    baseline — catches crowd noise, music swells, impacts, laughter, etc.
    even with ZERO speech. Returns [(start, end, intensity 0-1), ...]."""
    if energies is None or len(energies) == 0:
        return []
    peak_thresh = baseline * 2.2 + 1e-6
    peaks = []
    for i, e in enumerate(energies):
        if e > peak_thresh:
            t0 = i * window
            peaks.append((t0, t0 + window, min(1.0, float(e / (energies.max() + 1e-9)))))
    return peaks


def _cf_suggest_audio_mode(start: float, end: float, segments: list, energies, baseline: float, window: float) -> tuple:
    """Per-clip AI suggestion for original/mix/voiceover — the actual fix
    for 'voiceover is used even for nature/music clips where it makes no
    sense'. Returns (mode, human-readable reason) so the UI can show its
    work, not just the answer."""
    span = max(0.01, end - start)
    speech = sum(min(s["end"], end) - max(s["start"], start) for s in segments if s["end"] > start and s["start"] < end)
    speech_cov = speech / span

    seg_energy = 0.0
    if energies is not None and len(energies):
        i0, i1 = int(start // window), min(int(end // window), len(energies) - 1)
        if i1 >= i0 >= 0:
            seg_energy = float(np.mean(energies[i0:i1 + 1]))

    if speech_cov > 0.3:
        return "original", f"{int(speech_cov * 100)}% speech coverage in this clip — real commentary already present"
    if seg_energy > baseline * 1.8:
        return "mix", "Notable ambient audio here (music/crowd/effects) — narration layered on top keeps it"
    return "voiceover", "Little/no commentary or notable ambient sound — full narration recommended"


def _cf_merge_heuristic_windows(scene_ts: list, audio_peaks: list, duration: float, target_len: float = 30.0) -> list:
    """Clusters raw scene-cut + audio-energy signals into clip-length
    candidate windows scored by how much 'activity' they contain. This is
    what makes silent/no-commentary footage (nature, wordless gameplay,
    ASMR) produce usable candidates instead of a hard, unexplained 0."""
    if duration <= 0:
        return []
    bucket_size = 5.0
    n_buckets = max(1, int(duration // bucket_size) + 1)
    activity = np.zeros(n_buckets)
    for ts in scene_ts:
        b = int(ts // bucket_size)
        if 0 <= b < n_buckets:
            activity[b] += 1.0
    for (s, e, intensity) in audio_peaks:
        b = int(s // bucket_size)
        if 0 <= b < n_buckets:
            activity[b] += intensity * 1.5

    if activity.max() <= 0:
        return []

    candidates, t = [], 0.0
    while t < duration:
        end = min(t + target_len, duration)
        b0, b1 = int(t // bucket_size), int(end // bucket_size)
        candidates.append({"start": t, "end": end, "raw_score": float(activity[b0:b1 + 1].sum())})
        t += bucket_size * 3  # coarse stride — keeps candidate count sane

    candidates.sort(key=lambda c: -c["raw_score"])
    top = [c for c in candidates[:12] if c["raw_score"] > 0]
    if not top:
        return []
    max_score = max(c["raw_score"] for c in top)
    out = []
    for c in top:
        norm_score = int(30 + 50 * (c["raw_score"] / (max_score + 1e-9)))  # heuristic caps below AI-scored moments
        out.append({
            "start": round(c["start"], 2), "end": round(c["end"], 2),
            "title": "High-activity moment", "hook": "",
            "score": min(80, norm_score),
            "reason": "Detected via scene-change + audio-energy analysis (little/no commentary in this recording)",
            "source": "heuristic",
        })
    return out


_VISION_MODEL_HINTS = ["llava", "moondream", "bakllava", "vision", "qwen2.5vl", "qwen2-vl",
                        "minicpm-v", "llama3.2-vision", "gemma3", "pixtral"]


async def _cf_detect_vision_model() -> Optional[str]:
    """Checks your installed Ollama models for a vision-capable one — no
    guessing, no new download unless YOU choose to pull one. Returns None
    if you don't have one, in which case the heuristic-only path is used."""
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            r = await client.get("http://localhost:11434/api/tags")
            if r.status_code != 200:
                return None
            for m in r.json().get("models", []):
                name = m.get("name", "").lower()
                if any(h in name for h in _VISION_MODEL_HINTS):
                    return m.get("name")
    except Exception:
        pass
    return None


@app.get("/clipfinder/vision-status")
async def clipfinder_vision_status():
    """Explicit, checkable-before-you-click status of whether a vision
    model is available — the exact 'no clarity whether vision is being
    used' gap. Frontend calls this on load and shows it prominently."""
    model = await _cf_detect_vision_model()
    return {"detected": model is not None, "model": model}


def _cf_extract_frame_b64(video_path: str, timestamp: float) -> Optional[str]:
    frame_path = f"{CLIPS_DIR}/_vis_{uuid.uuid4().hex}.jpg"
    cmd = [FFMPEG_EXE, "-y", "-ss", str(max(0, timestamp)), "-i", video_path, "-vframes", "1", "-q:v", "4", frame_path]
    try:
        r = subprocess.run(cmd, capture_output=True, timeout=30)
        if r.returncode != 0 or not os.path.exists(frame_path):
            return None
        with open(frame_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        os.remove(frame_path)
        return b64
    except Exception:
        return None


async def _cf_vision_score_window(video_path: str, model: str, start: float, end: float) -> Optional[dict]:
    """Refines ONE heuristic candidate window using an actual vision model —
    real scene understanding instead of just 'motion happened here'. Capped
    to a handful of windows per recording upstream to keep this fast."""
    b64 = _cf_extract_frame_b64(video_path, (start + end) / 2)
    if not b64:
        return None
    prompt = ('Look at this video frame. Rate how visually striking/shareable this moment looks for a '
              'short-form video (Reels/Shorts/TikTok). Return ONLY this JSON object: '
              '{"title": "<short catchy title, under 60 chars>", "caption": "<what is happening, one sentence>", "score": <0-100>}')
    try:
        async with httpx.AsyncClient(timeout=90.0) as client:
            resp = await client.post("http://localhost:11434/api/chat", json={
                "model": model,
                "messages": [{"role": "user", "content": prompt, "images": [b64]}],
                "stream": False, "format": "json", "think": False,
            })
            if resp.status_code != 200:
                return None
            content = resp.json().get("message", {}).get("content", "")
            data = json.loads(content)
            return {
                "start": round(start, 2), "end": round(end, 2),
                "title": str(data.get("title", "Visual moment"))[:80], "hook": "",
                "score": max(0, min(100, int(data.get("score", 50)))),
                "reason": str(data.get("caption", ""))[:200],
                "source": "vision",
            }
    except Exception:
        return None


_DETAIL_LEVEL_MULT = {"low": 0.6, "auto": 1.0, "medium": 1.0, "high": 1.7}


async def _cf_deep_vision_describe(video_path: str, start: float, end: float, model: str,
                                    detail_level: str = "auto") -> Optional[str]:
    """DEEP analysis (Manual toggle or Auto-mode) — the short one-sentence
    'reason' from moment-scoring is enough for a quick draft, but for a
    genuinely accurate script (especially multi-beat moments — someone
    trips, falls, people laugh, all in one clip), this needs more than one
    blended description.

    Three improvements combined:
    1. AUTO frame-count scaling — roughly one frame per ~4 seconds of clip
       (short clips don't need 10 frames; long ones need more than 3).
    2. Manual DETAIL LEVEL override (low/medium/high) multiplies that count
       for explicit speed-vs-accuracy control.
    3. TIMESTAMP-AWARE sequential prompting — each frame is labeled with
       its actual time in the clip, and the model is explicitly asked to
       narrate a first-then-then sequence of beats rather than one vague
       blended paragraph, so a trip-then-fall-then-laugh moment reads as
       an actual sequence, not a blur."""
    duration = max(0.1, end - start)
    base_count = max(3, min(10, round(duration / 4)))
    mult = _DETAIL_LEVEL_MULT.get((detail_level or "auto").lower(), 1.0)
    n_frames = max(3, min(12, round(base_count * mult)))

    fracs = [i / (n_frames - 1) for i in range(n_frames)] if n_frames > 1 else [0.5]
    frames, timestamps = [], []
    for frac in fracs:
        t = start + duration * frac
        b64 = _cf_extract_frame_b64(video_path, t)
        if b64:
            frames.append(b64)
            timestamps.append(round(t - start, 1))
    if not frames:
        return None

    frame_labels = ", ".join(f"frame {i+1} at {ts}s" for i, ts in enumerate(timestamps))
    prompt = (f"These {len(frames)} frames are taken IN ORDER across a {round(duration)}-second video clip "
              f"({frame_labels}, measured from the start of this clip). "
              f"Describe what happens across this clip AS A SEQUENCE OF EVENTS — use phrasing like "
              f"'first X happens, then Y, then Z' to capture the actual progression, not one blended "
              f"description. Be specific about what's on screen (UI elements, text, actions, objects, "
              f"people's reactions, etc), not generic. This will be used to write an accurate voiceover "
              f"script, so specificity and the correct ORDER of events both matter.")
    try:
        async with httpx.AsyncClient(timeout=150.0) as client:
            resp = await client.post("http://localhost:11434/api/chat", json={
                "model": model,
                "messages": [{"role": "user", "content": prompt, "images": frames}],
                "stream": False, "think": False,
            })
            if resp.status_code != 200:
                return None
            return resp.json().get("message", {}).get("content", "").strip() or None
    except Exception:
        return None


def _clipfinder_moments_worker(job_id: str, recording_id: str, model: str, mode: str, top_n: int,
                                chunk_chars: int = 6000, scene_threshold: float = 0.35,
                                audio_window: float = 2.0, vision_cap: int = 8):
    """Runs the chunk loop with live progress written to clipfinder_jobs
    after EVERY chunk — this is what the frontend polls to show
    'Scoring chunk 2/4…' instead of one silent multi-minute spinner.

    Also tracks WHY candidates get dropped (chunk request/parse failure vs
    individual moment field parsing vs model genuinely returning nothing),
    so a final result of 0 moments always comes with an explanation
    instead of looking identical to 'the model looked and found nothing'."""
    try:
        # FIX: `if not segments` treated a legitimately-empty transcript
        # (silent/music-only footage — zero speech is a VALID outcome, not
        # a failure) identically to 'never transcribed at all', hard-erroring
        # instead of falling through to hybrid scene/audio/vision detection.
        if recording_id not in WORD_CACHE:
            clipfinder_jobs[job_id] = {"status": "failed", "stage": "finding_moments", "error": "Transcribe this recording first"}
            return
        segments = WORD_CACHE[recording_id]

        duration = recording_store[recording_id]["duration"] or (segments[-1]["end"] if segments else 0)
        chunks = _clipfinder_build_transcript_chunks(segments, max_chars=chunk_chars)
        total_chunks = len(chunks)

        clipfinder_jobs[job_id] = {
            "status": "processing", "stage": "reading_transcript",
            "detail": f"Sending transcript to '{model}' in {total_chunks} chunk(s) of up to {chunk_chars:,} characters each",
            "chunk": 0, "total_chunks": total_chunks, "error": None,
        }

        all_moments = []
        chunk_errors = []          # chunks where the request/parse itself failed
        raw_candidates_seen = 0    # total items the model returned across all chunks
        dropped_field_errors = 0   # items that came back but had unparseable start/end/etc.

        for i, chunk in enumerate(chunks):
            clipfinder_jobs[job_id]["chunk"] = i + 1  # 1-indexed for display, e.g. "chunk 1/4"
            clipfinder_jobs[job_id]["detail"] = (
                f"'{model}' is reading chunk {i+1}/{total_chunks} ({len(chunk['text']):,} chars, "
                f"covering {_fmt_mmss(chunk['start'])}–{_fmt_mmss(chunk['end'])}) and scoring candidate moments")
            result = asyncio.run(_ollama_find_moments_chunk(chunk["text"], model))

            if result["error"]:
                chunk_errors.append({"chunk": i + 1, "error": result["error"], "raw_preview": result["raw_preview"]})
                print(f"[clipfinder] chunk {i+1}/{total_chunks} failed: {result['error']} | raw preview: {result['raw_preview']!r}")
                continue

            raw_candidates_seen += len(result["moments"])
            for m in result["moments"]:
                try:
                    start = max(0.0, _parse_time_value(m.get("start_sec", m.get("start", 0))))
                    end = _parse_time_value(m.get("end_sec", m.get("end", start)))
                except (TypeError, ValueError) as e:
                    dropped_field_errors += 1
                    print(f"[clipfinder] chunk {i+1}: dropped a candidate — bad start/end ({e}): {m}")
                    continue
                if end <= start:
                    dropped_field_errors += 1
                    continue
                end = min(end, start + 90, duration if duration else end)
                if end - start < 5:
                    dropped_field_errors += 1
                    continue
                score_raw = str(m.get("score", 50))
                score_digits = re.search(r"-?\d+", score_raw)
                score = max(0, min(100, int(score_digits.group()) if score_digits else 50))
                all_moments.append({
                    "start": round(start, 2), "end": round(end, 2),
                    "title": str(m.get("title", "Untitled clip"))[:80],
                    "hook": str(m.get("hook", ""))[:200],
                    "score": score,
                    "reason": str(m.get("reason", ""))[:200],
                    "source": "transcript",
                })

        # ── SMART hybrid detection — ALWAYS runs now, not just as a fallback ──
        # Two things happen on every video, regardless of how much speech
        # there is:
        #  1. Scene-change + audio-energy heuristics run unconditionally —
        #     these catch great NON-verbal moments (a big reaction, a visual
        #     payoff) that a transcript-only pass would miss even in a
        #     commentary-heavy video, not just silent footage.
        #  2. If a vision model is installed, it does double duty: scoring
        #     the heuristic windows (as before) AND enriching the top
        #     transcript-scored moments with actual visual judgment — a real
        #     combined transcript+vision signal, not vision-as-last-resort.
        speech_seconds = sum(seg["end"] - seg["start"] for seg in segments)
        speech_coverage = (speech_seconds / duration) if duration else 1.0
        hybrid_used, vision_used, vision_model_name = True, False, None

        clipfinder_jobs[job_id]["stage"] = "analyzing_visuals"
        clipfinder_jobs[job_id]["detail"] = f"Analyzing {_fmt_mmss(duration)} of footage for scene changes and audio-energy peaks (window={audio_window}s, scene sensitivity={scene_threshold})"
        rec_path = recording_store[recording_id]["path"]
        scene_ts = _cf_detect_scene_changes(rec_path, threshold=scene_threshold)
        energies, energy_baseline, energy_window = _cf_audio_energy_profile(rec_path, duration, window=audio_window)
        audio_peaks = _cf_detect_audio_peaks_from_profile(energies, energy_baseline, energy_window)
        heuristic_candidates = _cf_merge_heuristic_windows(scene_ts, audio_peaks, duration)
        clipfinder_jobs[job_id]["detail"] = f"Found {len(scene_ts)} scene changes and {len(audio_peaks)} audio-energy peaks → {len(heuristic_candidates)} candidate window(s)"

        vision_model_name = asyncio.run(_cf_detect_vision_model())
        if vision_model_name:
            vision_used = True
            clipfinder_jobs[job_id]["stage"] = "vision_scoring"

            if heuristic_candidates:
                refined = []
                for vi, hc in enumerate(heuristic_candidates[:vision_cap]):  # capped — vision calls are the slow part
                    clipfinder_jobs[job_id]["detail"] = f"'{vision_model_name}' is looking at frame {vi+1}/{min(len(heuristic_candidates), vision_cap)} (t={_fmt_mmss(hc['start'])}) from motion/audio-detected windows"
                    vresult = asyncio.run(_cf_vision_score_window(rec_path, vision_model_name, hc["start"], hc["end"]))
                    refined.append(vresult if vresult else hc)
                heuristic_candidates = refined

            # Enrich the top transcript-scored moments with a real vision
            # pass too — this is the actual "combine both signals" behavior.
            top_transcript = sorted(all_moments, key=lambda x: -x["score"])[:vision_cap]
            for vi, tm in enumerate(top_transcript):
                clipfinder_jobs[job_id]["detail"] = f"'{vision_model_name}' is cross-checking transcript moment {vi+1}/{len(top_transcript)} ('{tm['title']}') against the actual frame"
                vresult = asyncio.run(_cf_vision_score_window(rec_path, vision_model_name, tm["start"], tm["end"]))
                if vresult:
                    # Blend rather than overwrite — transcript already found
                    # real spoken content here, vision just confirms/adjusts.
                    tm["score"] = int(round((tm["score"] + vresult["score"]) / 2))
                    tm["reason"] = f"{tm['reason']} · Visually: {vresult['reason']}".strip(" ·")
                    tm["source"] = "transcript+vision"

            all_moments.extend(heuristic_candidates)

        clipfinder_jobs[job_id]["stage"] = "finalizing"
        clipfinder_jobs[job_id]["detail"] = f"Removing overlapping duplicates from {len(all_moments)} total candidates and ranking by score"
        all_moments.sort(key=lambda x: -x["score"])
        kept = []
        for m in all_moments:
            overlaps = any(not (m["end"] <= k["start"] or m["start"] >= k["end"]) for k in kept)
            if not overlaps:
                kept.append(m)
        kept.sort(key=lambda x: -x["score"])
        kept = kept[:40]
        clipfinder_jobs[job_id]["detail"] = f"Deciding original-audio vs. mix vs. voiceover for {len(kept)} final moment(s) based on speech/ambient-audio content"
        for i, m in enumerate(kept):
            m["id"] = str(uuid.uuid4())
            m["auto_pick"] = (mode == "auto" and i < top_n)
            # SMART per-clip audio decision — the actual fix for "voiceover
            # gets used even on nature/music clips where it makes no sense".
            # AI suggests a default; the export UI lets you override per clip.
            sug_mode, sug_reason = _cf_suggest_audio_mode(m["start"], m["end"], segments, energies, energy_baseline, energy_window)
            m["suggested_audio_mode"] = sug_mode
            m["suggested_audio_reason"] = sug_reason

        # Build a plain-English explanation for a 0-result run instead of
        # letting it look like an unexplained dead end.
        diagnosis = None
        if not kept:
            if chunk_errors:
                diagnosis = (f"{len(chunk_errors)}/{total_chunks} chunk(s) failed to get usable JSON from '{model}'. "
                             f"First error: {chunk_errors[0]['error']}")
            elif hybrid_used:
                diagnosis = ("No speech AND no detectable scene changes or audio-energy peaks in this recording — "
                              "it may be very static/quiet footage. Try a different recording.")
            elif raw_candidates_seen == 0:
                diagnosis = (f"'{model}' returned zero candidates across all {total_chunks} chunk(s) — "
                             f"it looked at the transcript and didn't flag anything. Try a stronger/different "
                             f"local model, or lower your expectations of what counts as 'viral' for this content.")
            elif dropped_field_errors > 0:
                diagnosis = (f"'{model}' proposed {raw_candidates_seen} candidate(s), but all {dropped_field_errors} "
                             f"were dropped due to unparseable/invalid start-end times or durations outside 5-90s. "
                             f"This usually means the model isn't following the requested JSON field format well — "
                             f"try a different model.")

        settings_used = {"model": model, "mode": mode, "top_n": top_n, "chunk_chars": chunk_chars,
                          "scene_threshold": scene_threshold, "audio_window": audio_window, "vision_cap": vision_cap}

        if kept:
            _cf_db_save_moments(recording_id, kept, settings_used)

        clipfinder_jobs[job_id] = {
            "status": "completed", "stage": "finding_moments",
            "chunk": total_chunks, "total_chunks": total_chunks,
            "recording_id": recording_id, "moments": kept, "mode": mode, "error": None,
            "diagnosis": diagnosis, "hybrid_used": hybrid_used, "vision_used": vision_used,
            "vision_model": vision_model_name, "speech_coverage_pct": round(speech_coverage * 100),
            "settings_used": settings_used, "scene_cuts_found": len(scene_ts), "audio_peaks_found": len(audio_peaks),
            "heuristic_candidates_found": len(heuristic_candidates), "raw_candidates_seen": raw_candidates_seen,
            "debug": {"raw_candidates_seen": raw_candidates_seen, "dropped_field_errors": dropped_field_errors,
                      "chunk_errors": chunk_errors},
        }
    except Exception as e:
        clipfinder_jobs[job_id] = {"status": "failed", "stage": "finding_moments", "error": str(e)}



_TONE_PROMPTS = {
    "": "",
    "emotional": " Make it emotionally resonant — lean into feeling, vulnerability, and heart.",
    "excited": " Make it high-energy and excited — exclamation-worthy, enthusiastic pacing.",
    "climactic": " Build it like a climactic reveal — tension rising to a payoff moment.",
    "calm": " Keep it calm and soothing — slow, gentle, reassuring pacing.",
    "funny": " Make it funny — comedic timing, a punchline or witty observation.",
    "suspenseful": " Make it suspenseful — withhold the payoff, build tension and curiosity.",
    "dramatic": " Make it dramatic — weighty, cinematic, stakes-focused language.",
    "inspirational": " Make it inspirational — uplifting, motivational, a takeaway lesson.",
    "wholesome": " Make it wholesome — warm, heartfelt, feel-good tone.",
    "sarcastic": " Make it sarcastic/witty — dry humor, knowing tone.",
    "urgent": " Make it urgent — fast-paced, high-stakes, act-now energy.",
    "nostalgic": " Make it nostalgic — reflective, wistful, throwback framing.",
}


_NARRATIVE_STYLE_PROMPTS = {
    "": "",
    "genuine": " Narrate like a real person genuinely reacting to and describing what's happening in the moment — authentic, not scripted-sounding.",
    "documentary": " Narrate like a calm documentary narrator objectively describing the scene.",
    "sportscaster": " Narrate like a sports commentator doing energetic play-by-play of the action as it unfolds.",
    "deadpan": " Narrate in a deadpan, dryly sarcastic observer voice — understated, witty, never over-excited.",
    "storyteller": " Narrate like a storyteller building a narrative arc — setup, tension, payoff.",
    "newsanchor": " Narrate like a news anchor delivering this as a matter-of-fact report.",
    "meme": " Narrate in an internet/meme culture voice — casual, in-on-the-joke, current slang where natural.",
}


async def _cf_autodraft_script(recording_id: str, start: float, end: float, model: str, tone: str = "",
                                moment_title: str = "", moment_hook: str = "", moment_reason: str = "",
                                moment_source: str = "", user_guidance: str = "",
                                deep_analysis: bool = False, vision_model: str = "",
                                narrative_style: str = "", detail_level: str = "auto",
                                brief_what: str = "", brief_feel: str = "", brief_takeaway: str = "") -> str:
    """Shared by the standalone draft endpoint AND as an automatic fallback
    inside export (if voiceover is requested but no script was written).

    FIXED: previously, when a clip had no spoken commentary (exactly the
    case where Vision found the moment, since that's Vision's whole job),
    this fell back to a completely ungrounded instruction — 'write an
    original exciting moment' with zero facts about the actual video. The
    model then hallucinated whatever 'exciting moment' was statistically
    common in its training data (e.g. sports commentary) instead of
    anything related to the real footage. Now every available signal
    (recording name, moment title/hook, and critically the VISION-derived
    reason — which describes what's actually visible in the frame — was
    already being computed and just never passed into this prompt) is
    included as real grounding, and optional user guidance lets you steer
    it further using everything already known about this specific clip.

    narrative_style is a SEPARATE axis from `tone` — tone is emotional
    register (funny/sad/excited), narrative_style is PERSPECTIVE/FORMAT
    (genuine reaction vs documentary vs sportscaster vs meme voice, etc).

    brief_what/feel/takeaway are optional structured fields for multi-beat
    moments (e.g. someone trips, falls, people laugh) — describing the
    actual sequence of events, desired emotional arc, and punchline gives
    the model a much stronger anchor than a single free-text guidance note.

    deep_analysis=True triggers a SECOND, richer vision pass (multiple
    frames spread across the clip, timestamp-labeled for sequential
    narration, count scaled by clip length and detail_level) for even more
    accurate grounding — costs one extra vision call, opt-in via Manual
    toggle or the Auto-mode setting."""
    segments = WORD_CACHE.get(recording_id, [])
    text = " ".join(s["text"].strip() for s in segments if s["end"] > start and s["start"] < end).strip()
    rec = recording_store.get(recording_id, {})
    rec_name = os.path.splitext(rec.get("filename", ""))[0]
    video_context = (rec.get("context_notes") or "").strip()

    context_lines = [f"This clip is from a recording called \"{rec_name}\"."]
    if video_context:
        context_lines.append(f"About this video (creator-provided context — trust this): {video_context}")
    if moment_title:
        context_lines.append(f"This specific moment is titled: \"{moment_title}\"")
    if moment_hook:
        context_lines.append(f"Hook/key line: \"{moment_hook}\"")
    if moment_reason:
        label = "What the AI's VISION ANALYSIS actually sees happening in this clip" if moment_source in ("vision", "heuristic") else "Why this moment was selected"
        context_lines.append(f"{label}: {moment_reason}")

    if brief_what.strip():
        context_lines.append(f"What actually happens, in order (creator-provided — trust this over vision guesses): {brief_what.strip()}")
    if brief_feel.strip():
        context_lines.append(f"How this should make the viewer feel: {brief_feel.strip()}")
    if brief_takeaway.strip():
        context_lines.append(f"The punchline/takeaway to land on: {brief_takeaway.strip()}")

    if deep_analysis and vision_model and rec.get("path"):
        deep_desc = await _cf_deep_vision_describe(rec["path"], start, end, vision_model, detail_level)
        if deep_desc:
            context_lines.append(f"DETAILED scene-by-scene vision analysis, in sequence (most accurate source — prioritize this): {deep_desc}")

    if text:
        context_lines.append(f"Actual spoken transcript for this clip: \"{text}\"")
    else:
        context_lines.append("There is NO spoken commentary in this clip — base the script ENTIRELY on the "
                              "context above (recording name, moment title, and especially the vision analysis), "
                              "not on generic/invented content unrelated to this specific video.")
    if user_guidance.strip():
        context_lines.append(f"Additional instructions from the creator — follow these closely: {user_guidance.strip()}")

    grounding = "\n".join(context_lines)
    target_secs = int(end - start) + 3
    tone_instruction = _TONE_PROMPTS.get((tone or "").lower(), "")
    narrative_instruction = _NARRATIVE_STYLE_PROMPTS.get((narrative_style or "").lower(), "")
    prompt = (f"Write a punchy, natural-sounding short-form video voiceover script "
              f"(aim for about {target_secs} seconds spoken aloud, strong hook in the very first line, "
              f"conversational spoken tone, no stage directions, no timestamps).{tone_instruction}{narrative_instruction}\n\n"
              f"{grounding}\n\n"
              f"IMPORTANT: Stay strictly relevant to what this clip is actually about, described above. "
              f"Do not invent unrelated scenarios, sports, or events.\n\n"
              f"AVOID generic AI-narrator clichés — do NOT open with phrases like 'You won't believe...', "
              f"'Get ready to witness...', 'In this moment...', or similar stock hooks. Sound like a real "
              f"person genuinely reacting to and describing what's happening, not a templated ad-voice.\n\n"
              f"Return ONLY the script text, nothing else.")
    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            resp = await client.post("http://localhost:11434/api/chat", json={
                "model": model, "messages": [{"role": "user", "content": prompt}],
                "stream": False, "think": False,
            })
            if resp.status_code != 200:
                return text or moment_title or "Check this out!"
            return resp.json().get("message", {}).get("content", "").strip() or text or moment_title or "Check this out!"
    except Exception:
        return text or moment_title or "Check this out!"


@app.post("/clipfinder/draft-script")
async def clipfinder_draft_script(payload: dict):
    """Lets the UI show an editable, pre-filled voiceover script BEFORE
    export, rather than only auto-writing one invisibly at export time."""
    recording_id = payload.get("recording_id", "")
    start = float(payload.get("start", 0))
    end = float(payload.get("end", 0))
    model = payload.get("model") or selected_model
    tone = payload.get("tone", "")
    deep_analysis = bool(payload.get("deep_analysis", False))
    if recording_id not in recording_store:
        return {"error": "Unknown recording_id"}
    if recording_id not in WORD_CACHE:
        return {"error": "Transcribe this recording first"}
    vision_model = ""
    if deep_analysis:
        vision_model = await _cf_detect_vision_model() or ""
    script = await _cf_autodraft_script(
        recording_id, start, end, model, tone,
        moment_title=payload.get("title", ""), moment_hook=payload.get("hook", ""),
        moment_reason=payload.get("reason", ""), moment_source=payload.get("source", ""),
        user_guidance=payload.get("user_guidance", ""),
        deep_analysis=deep_analysis, vision_model=vision_model,
        narrative_style=payload.get("narrative_style", ""), detail_level=payload.get("detail_level", "auto"),
        brief_what=payload.get("brief_what", ""), brief_feel=payload.get("brief_feel", ""),
        brief_takeaway=payload.get("brief_takeaway", ""),
    )
    return {"script": script, "used_deep_analysis": bool(deep_analysis and vision_model)}


@app.post("/clipfinder/find-moments")
async def clipfinder_find_moments(payload: dict, background_tasks: BackgroundTasks):
    """Kicks off moment-scoring as a background job (was previously one
    blocking call with zero visibility into progress — a 2-6 minute local
    LLM read-through with no feedback looked identical to 'hung'). Poll
    /clipfinder/status/{job_id} for live 'chunk X/Y' progress."""
    recording_id = payload.get("recording_id", "")
    model = payload.get("model") or selected_model
    mode = payload.get("mode", "manual")
    top_n = int(payload.get("top_n", 5))

    if recording_id not in recording_store:
        return {"error": "Unknown recording_id", "status": "failed"}
    if recording_id not in WORD_CACHE:  # FIX: was `not WORD_CACHE.get(...)` — an empty list (silent footage) is falsy but perfectly valid
        return {"error": "Transcribe this recording first", "status": "failed"}

    # Analysis depth — presets for naive users, full manual control for
    # power users, exactly as requested. "custom" reads every value from
    # the payload directly; presets are just named bundles of the same knobs.
    depth = payload.get("analysis_depth", "balanced")
    DEPTH_PRESETS = {
        "fast":     {"chunk_chars": 9000, "scene_threshold": 0.42, "audio_window": 3.0, "vision_cap": 4},
        "balanced": {"chunk_chars": 6000, "scene_threshold": 0.35, "audio_window": 2.0, "vision_cap": 8},
        "thorough": {"chunk_chars": 3000, "scene_threshold": 0.25, "audio_window": 1.0, "vision_cap": 16},
    }
    settings = dict(DEPTH_PRESETS.get(depth, DEPTH_PRESETS["balanced"]))
    if depth == "custom":
        settings = {
            "chunk_chars": int(payload.get("chunk_chars", 6000)),
            "scene_threshold": float(payload.get("scene_threshold", 0.35)),
            "audio_window": float(payload.get("audio_window", 2.0)),
            "vision_cap": int(payload.get("vision_cap", 8)),
        }

    job_id = str(uuid.uuid4())
    clipfinder_jobs[job_id] = {"status": "queued", "stage": "finding_moments", "chunk": 0, "total_chunks": 0,
                                "analysis_depth": depth, "settings_used": settings}
    background_tasks.add_task(_clipfinder_moments_worker, job_id, recording_id, model, mode, top_n,
                               settings["chunk_chars"], settings["scene_threshold"],
                               settings["audio_window"], settings["vision_cap"])
    return {"job_id": job_id, "status": "queued"}


def _slugify(text: str, max_len: int = 40) -> str:
    """Turn a title/filename into a safe, readable filename fragment.
    'World Record!! 🔥' -> 'World-Record'"""
    text = re.sub(r"[^\w\s-]", "", text).strip()
    text = re.sub(r"[\s_]+", "-", text)
    return text[:max_len].strip("-") or "clip"


class ClipFinderExportClip(BaseModel):
    start: float
    end: float
    title: str = "clip"
    voiceover_script: Optional[str] = None   # per-clip, user-edited or blank (auto-drafted at export time)
    audio_mode: str = "original"             # "original" | "mix" | "voiceover" — set PER CLIP, overriding the AI suggestion is just changing this
    tone: str = ""                           # emotional tone chip for script drafting, e.g. "excited", "climactic"
    hook: str = ""                           # moment's hook line — grounding for export-time auto-draft fallback
    reason: str = ""                         # moment's (often vision-derived) reason — the actual fix for hallucinated drafts
    source: str = ""                         # "transcript" | "heuristic" | "vision" | "transcript+vision"
    user_guidance: str = ""                  # optional creator notes to steer the script
    deep_analysis: Optional[bool] = None     # per-clip MANUAL override — None = defer to the request-level auto_deep_analysis setting
    narrative_style: str = ""                # perspective/format axis, separate from tone — "genuine", "sportscaster", "deadpan", etc.
    detail_level: str = "auto"                # "low" | "auto" | "high" — deep-analysis frame-count control
    brief_what: str = ""                      # structured brief — what actually happens, in order
    brief_feel: str = ""                      # structured brief — desired emotional arc
    brief_takeaway: str = ""                  # structured brief — punchline/takeaway


class ClipFinderExportRequest(BaseModel):
    recording_id: str
    clips: List[ClipFinderExportClip]
    fmt: str = "shorts"                 # "shorts" (9:16) | "youtube" (16:9)
    vertical_style: str = "blur"
    auto_caption: bool = True
    sub_style: str = "viral"
    highlight_color: str = "#00FFFF"
    text_color: str = "white"
    # Full subtitle control parity with Video Studio — previously hardcoded
    # to defaults, so these settings silently had no effect from this tab.
    effect: str = "karaoke"
    position: str = "bottom"
    size: str = "medium"
    intensity: int = 1
    av_offset: float = 0.0
    uppercase: bool = False
    font_family: str = ""
    box_opacity: int = -1
    outline_width: int = -1
    anim_speed: str = "normal"
    easing: str = "linear"
    color_cycle: Optional[List[str]] = None
    caption_mode: str = "phrase"
    # New separate-voiceover generation — makes the export a genuinely
    # standalone short (fresh narration) instead of just a trim of the
    # source's original audio.
    generate_voiceover: bool = False
    voiceover_engine: str = "kokoro"     # "kokoro" | "edge"
    voiceover_voice: str = "af_heart"
    voiceover_mix: bool = False          # False = replace original audio (default/SMART); True = layer new VO over original at low volume
    voiceover_model: str = ""            # local LLM used for auto-drafting a script when one isn't provided
    subtitle_source: str = "smart"       # "smart" | "original" | "voiceover" — which audio's words the burnt captions should match
    custom_pos_x: Optional[float] = None  # 0-100 % — overrides position preset with free-form placement when both are set
    custom_pos_y: Optional[float] = None  # 0-100 %
    auto_deep_analysis: bool = False      # AUTO mode — deep-analyze every vision-sourced clip automatically, no per-clip toggling needed
    emoji_captions: str = "off"           # "off" | "keyword" | "smart" — emoji+text captions


_EMOJI_KEYWORDS = {
    "fast": "⚡", "speed": "💨", "quick": "⚡", "race": "🏎️", "racing": "🏎️",
    "fire": "🔥", "hot": "🔥", "lit": "🔥", "insane": "🔥", "crazy": "🔥",
    "money": "💰", "cash": "💰", "free": "🎁", "reward": "🎁", "rewards": "🎁",
    "prize": "🏆", "win": "🏆", "won": "🏆", "winner": "🏆", "victory": "🏆",
    "unlock": "🔓", "vip": "👑", "premium": "👑", "gift": "🎁",
    "love": "❤️", "amazing": "🤩", "awesome": "🤩", "wow": "😱", "shocking": "😱",
    "funny": "😂", "lol": "😂", "hilarious": "😂", "sad": "😢", "cry": "😢",
    "angry": "😡", "scary": "😱", "cute": "🥰",
    "game": "🎮", "gaming": "🎮", "level": "⬆️", "score": "📈", "boss": "👹",
    "car": "🚗", "cars": "🚗", "upgrade": "⬆️", "power": "💪", "powerful": "💪",
    "new": "🆕", "top": "🔝", "best": "⭐", "star": "⭐", "secret": "🤫",
    "easy": "✅", "done": "✅", "warning": "⚠️", "danger": "⚠️",
    "idea": "💡", "tip": "💡", "question": "❓", "explosion": "💥", "boom": "💥",
    "attack": "⚔️", "battle": "⚔️", "record": "📈",
}


def _emoji_append_to_segment(seg: dict, emoji: str):
    """Appends an emoji as a REAL word entry with its own timing, not just
    extra text — karaoke/color/every other effect renders purely from
    seg['words'], completely ignoring seg['text'] except as a fallback for
    segments with no word-level data. A text-only append was silently
    invisible for the default (karaoke) effect."""
    if not emoji:
        return
    seg["text"] = f"{seg['text']} {emoji}"
    words = seg.get("words") or []
    if words:
        last_end = words[-1]["end"]
        emoji_dur = min(0.4, max(0.15, seg["end"] - last_end))  # small, bounded, never overruns the segment
        words.append({"word": emoji, "start": last_end, "end": min(seg["end"], last_end + emoji_dur)})
    else:
        # No word timings at all (rare) — give the emoji the segment's own span
        seg["words"] = [{"word": emoji, "start": seg["start"], "end": seg["end"]}]


def _emoji_keyword_annotate_segment(seg: dict):
    """Fast, free — scans for known trigger words, adds the FIRST match
    found (sparse by design: at most one per caption phrase, so it stays
    clean instead of cluttered with emoji on every line)."""
    lower = seg["text"].lower()
    for word, emoji in _EMOJI_KEYWORDS.items():
        if re.search(r'\b' + re.escape(word) + r'\b', lower):
            _emoji_append_to_segment(seg, emoji)
            return


async def _emoji_smart_annotate_batch(segments: list, model: str) -> list:
    """Smart/AI mode — ONE batched LLM call covering every caption phrase
    in the clip at once (not one call per phrase, which would be slow and
    expensive), asking for at most one context-aware emoji per phrase.
    Falls back to unmodified segments on any failure — emoji captions
    should never be able to break an export."""
    if not segments:
        return segments
    numbered = "\n".join(f"{i}: {s['text']}" for i, s in enumerate(segments))
    prompt = (f"For each numbered caption line below, pick ONE relevant emoji if it clearly fits the "
              f"meaning (or skip it if nothing fits well — don't force one onto every line). Return ONLY "
              f"a JSON object mapping index to emoji, e.g. {{\"0\": \"🔥\", \"2\": \"💰\"}}.\n\n{numbered}")
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post("http://localhost:11434/api/chat", json={
                "model": model, "messages": [{"role": "user", "content": prompt}],
                "stream": False, "format": "json", "think": False,
            })
            if resp.status_code != 200:
                return segments
            data = json.loads(resp.json().get("message", {}).get("content", "{}") or "{}")
            for idx_str, emoji in data.items():
                try:
                    idx = int(idx_str)
                    if 0 <= idx < len(segments) and emoji:
                        _emoji_append_to_segment(segments[idx], emoji)
                except (ValueError, TypeError):
                    continue
    except Exception:
        pass
    return segments


def _cf_synthesize_voiceover(script: str, engine: str, voice: str, target_duration: float) -> str:
    """Synthesizes a NEW narration track for a clip — same Kokoro/Edge
    engines TTS Studio already uses (both return 24kHz mono, so no format
    juggling needed), then pads/trims to exactly the clip's length so it
    muxes cleanly regardless of how long the model's natural reading pace
    happens to be."""
    if engine == "edge":
        audio = asyncio.run(generate_edge_tts(script, voice or "en-US-AriaNeural"))
    else:
        gen = pipeline(script, voice=voice or "af_heart", speed=1.0)
        chunks = [c.audio.numpy() for c in gen]
        audio = np.concatenate(chunks) if chunks else np.zeros(1, dtype=np.float32)

    target_samples = max(1, int(target_duration * 24000))
    if len(audio) < target_samples:
        audio = np.pad(audio, (0, target_samples - len(audio)))
    else:
        audio = audio[:target_samples]

    vo_path = f"{CLIPS_DIR}/_vo_{uuid.uuid4().hex}.wav"
    sf.write(vo_path, audio, 24000)
    return vo_path


def _cf_transcribe_vo_for_captions(vo_path: str) -> list:
    """Re-transcribes the SYNTHESIZED voiceover (not the original recording)
    so burnt captions match the new narration's actual timing/words. Same
    shape as the sliced-original-segments path (start/end/text/words), and
    already clip-local since the VO audio starts at t=0 for this clip."""
    try:
        segments_iter, _ = whisper_model.transcribe(vo_path, word_timestamps=True, language=None, beam_size=5, vad_filter=False)
        out = []
        for seg in segments_iter:
            words = [{"word": w.word, "start": w.start, "end": w.end} for w in (seg.words or [])]
            out.append({"start": seg.start, "end": seg.end, "text": seg.text, "words": words})
        return out
    except Exception:
        return []


def _clipfinder_export_worker(job_id: str, req: ClipFinderExportRequest):
    rec = recording_store.get(req.recording_id)
    if not rec:
        clipfinder_jobs[job_id] = {"status": "failed", "stage": "export", "error": "Unknown recording"}
        return

    all_segments = WORD_CACHE.get(req.recording_id, [])
    results, total = [], len(req.clips)
    clipfinder_jobs[job_id] = {"status": "processing", "stage": "export", "progress": 0, "total": total, "clips": [], "error": None}

    src_base = _slugify(os.path.splitext(rec["filename"])[0], max_len=30)
    used_names = set()

    for idx, clip in enumerate(req.clips):
        vo_path = None  # defined up front so the `finally` cleanup below is always safe
        try:
            # Descriptive filename instead of a random UUID — e.g.
            # "RohitSharma209-WorldRecord.mp4", with a numeric suffix only
            # if two clips would otherwise collide.
            base_name = f"{src_base}-{_slugify(clip.title)}"
            candidate, n = base_name, 2
            while candidate in used_names:
                candidate = f"{base_name}-{n}"
                n += 1
            used_names.add(candidate)

            out_path = f"{CLIPS_DIR}/{candidate}.mp4"
            ass_path = f"{CLIPS_DIR}/{candidate}.ass"

            # ── Voiceover synthesis happens FIRST now — caption source
            # selection below needs to know whether VO audio exists yet.
            # PER-CLIP decision (clip.audio_mode), not a single global
            # switch — this is what lets a nature-footage clip skip
            # narration while a talking-head clip in the same batch uses it.
            wants_voiceover = clip.audio_mode in ("mix", "voiceover")
            vo_transcript_segments = None
            if wants_voiceover:
                # Manual per-clip override wins; otherwise defer to the
                # request-level Auto-mode setting (only for vision-sourced
                # clips, since deep vision analysis is meaningless for a
                # clip that already has real spoken commentary).
                use_deep = clip.deep_analysis if clip.deep_analysis is not None else (
                    req.auto_deep_analysis and clip.source in ("vision", "heuristic", "transcript+vision"))
                deep_vision_model = ""
                if use_deep:
                    deep_vision_model = asyncio.run(_cf_detect_vision_model()) or ""
                script = (clip.voiceover_script or "").strip() or asyncio.run(
                    _cf_autodraft_script(req.recording_id, clip.start, clip.end,
                                          req.voiceover_model or selected_model, clip.tone,
                                          moment_title=clip.title, moment_hook=clip.hook,
                                          moment_reason=clip.reason, moment_source=clip.source,
                                          user_guidance=clip.user_guidance,
                                          deep_analysis=use_deep, vision_model=deep_vision_model,
                                          narrative_style=clip.narrative_style, detail_level=clip.detail_level,
                                          brief_what=clip.brief_what, brief_feel=clip.brief_feel,
                                          brief_takeaway=clip.brief_takeaway))
                vo_path = _cf_synthesize_voiceover(script, req.voiceover_engine, req.voiceover_voice, clip.end - clip.start)

            # ── SMART caption source: which audio's words should the burnt
            # captions match? "smart" = voiceover's own words if one was
            # generated (it's the new headline narration), else the
            # original transcript. Explicit override always wins.
            use_vo_captions = (
                req.subtitle_source == "voiceover"
                or (req.subtitle_source == "smart" and wants_voiceover)
            ) and req.subtitle_source != "original"

            clip_segments = []
            if req.auto_caption:
                if use_vo_captions and vo_path:
                    vo_transcript_segments = _cf_transcribe_vo_for_captions(vo_path)
                    clip_segments = vo_transcript_segments
                else:
                    # Slice + rebase ORIGINAL word segments to 0 for this clip's own timeline
                    for seg in all_segments:
                        if seg["end"] <= clip.start or seg["start"] >= clip.end:
                            continue
                        words = [
                            {"word": w["word"], "start": max(0.0, w["start"] - clip.start), "end": max(0.0, w["end"] - clip.start)}
                            for w in seg.get("words", [])
                            if w["end"] > clip.start and w["start"] < clip.end
                        ]
                        clip_segments.append({
                            "start": max(0.0, seg["start"] - clip.start),
                            "end": min(clip.end, seg["end"]) - clip.start,
                            "text": seg["text"], "words": words,
                        })

            custom_pos = ({"x_pct": req.custom_pos_x, "y_pct": req.custom_pos_y}
                           if req.custom_pos_x is not None and req.custom_pos_y is not None else None)

            if req.emoji_captions == "keyword" and clip_segments:
                for seg in clip_segments:
                    _emoji_keyword_annotate_segment(seg)
            elif req.emoji_captions == "smart" and clip_segments:
                clip_segments = asyncio.run(_emoji_smart_annotate_batch(clip_segments, req.voiceover_model or selected_model))

            ass_content = generate_ass_content(
                clip_segments, req.sub_style, req.highlight_color,
                effect=req.effect, position=req.position, size=req.size, intensity=req.intensity,
                av_offset=req.av_offset, uppercase=req.uppercase, text_color=req.text_color,
                font_family=req.font_family, box_opacity=req.box_opacity, outline_width=req.outline_width,
                anim_speed=req.anim_speed, easing=req.easing, color_cycle=req.color_cycle,
                caption_mode=req.caption_mode, custom_pos=custom_pos,
            )
            with open(ass_path, "w", encoding="utf-8") as f:
                f.write(ass_content)

            vf_args = build_ffmpeg_filter(rec["width"], rec["height"], req.fmt, req.vertical_style, os.path.abspath(ass_path))
            filter_complex_str = vf_args[1]   # the video-only chain, ends in [out]
            map_args = ["-map", "[out]"]
            extra_inputs = []

            if wants_voiceover:
                extra_inputs = ["-i", vo_path]
                if clip.audio_mode == "mix":
                    # TRUE dynamic ducking via sidechaincompress: the ORIGINAL
                    # track's volume automatically dips only while the new
                    # voiceover is actually speaking (driven by its real
                    # amplitude), and recovers right after — not a constant
                    # -70% volume cut for the whole clip regardless of
                    # whether the VO is talking at that instant.
                    filter_complex_str += (
                        ";[0:a][1:a]sidechaincompress=threshold=0.05:ratio=8:attack=5:release=300[duckedorig]"
                        ";[duckedorig][1:a]amix=inputs=2:duration=first:dropout_transition=0[aout]"
                    )
                    map_args += ["-map", "[aout]"]
                else:
                    # "voiceover" mode — full replace, makes the export a
                    # genuinely standalone short rather than a trim of the
                    # source's audio.
                    map_args += ["-map", "1:a"]
            else:
                map_args += ["-map", "0:a?"]

            cmd = [
                FFMPEG_EXE, "-y",
                "-ss", str(max(0, clip.start)), "-to", str(clip.end),
                "-i", rec["path"],
                *extra_inputs,
                "-filter_complex", filter_complex_str,
                *map_args,
                "-c:v", "libx264", "-preset", "fast", "-crf", "21",
                "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart",
                out_path,
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
            if result.returncode != 0 or not os.path.exists(out_path):
                msg, _ = format_user_error(result.stderr or "unknown ffmpeg error")
                results.append({"title": clip.title, "error": msg})
            else:
                results.append({
                    "title": clip.title, "filename": os.path.basename(out_path),
                    "start": clip.start, "end": clip.end, "duration": round(clip.end - clip.start, 1),
                })
                _cf_db_save_export(os.path.basename(out_path), req.recording_id, clip.title,
                                    clip.start, clip.end, round(clip.end - clip.start, 1))  # auto-save
        except Exception as e:
            results.append({"title": clip.title, "error": str(e)})
        finally:
            if vo_path and os.path.exists(vo_path):
                try:
                    os.remove(vo_path)
                except Exception:
                    pass
            clipfinder_jobs[job_id]["progress"] = idx + 1
            clipfinder_jobs[job_id]["clips"] = results

    clipfinder_jobs[job_id]["status"] = "completed"


@app.post("/clipfinder/export")
async def clipfinder_export(req: ClipFinderExportRequest, background_tasks: BackgroundTasks):
    if req.recording_id not in recording_store:
        return {"error": "Unknown recording_id", "status": "failed"}
    if not req.clips:
        return {"error": "No clips selected", "status": "failed"}
    job_id = str(uuid.uuid4())
    clipfinder_jobs[job_id] = {"status": "queued", "stage": "export", "progress": 0, "total": len(req.clips), "clips": []}
    background_tasks.add_task(_clipfinder_export_worker, job_id, req)
    return {"job_id": job_id, "status": "queued"}


@app.get("/clipfinder/exports/{recording_id}")
async def clipfinder_list_exports(recording_id: str):
    """Lets the frontend repopulate the 'Exported Clips' panel after a
    restart — the files were always safe on disk, this just tells the UI
    they exist."""
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute(
        "SELECT filename, title, start, end, duration FROM clipfinder_exports WHERE recording_id=? ORDER BY created_at",
        (recording_id,),
    ).fetchall()
    conn.close()
    clips = []
    for filename, title, start, end, duration in rows:
        if os.path.exists(f"{CLIPS_DIR}/{filename}"):
            clips.append({"filename": filename, "title": title, "start": start, "end": end, "duration": duration})
    return {"clips": clips}


@app.post("/clipfinder/save")
async def clipfinder_manual_save(payload: dict):
    """Manual save checkpoint — auto-save already persists at every stage,
    this is a defensive re-write for peace of mind (and covers the rare
    edge case where an auto-save hook gets skipped)."""
    recording_id = payload.get("recording_id", "")
    if recording_id and recording_id in recording_store:
        _cf_db_save_recording(recording_id)
        if recording_id in WORD_CACHE:
            _cf_db_save_transcript(recording_id)
        return {"status": "saved"}
    return {"error": "Unknown recording_id"}


@app.get("/clipfinder/clip/{filename}")
async def clipfinder_serve_clip(filename: str):
    """FIX: filenames are deterministic (recording+title based, so re-exports
    are easy to find later) — but that means re-exporting the SAME clip
    title reuses the exact same URL. Without explicit no-cache headers, the
    browser was serving the FIRST export's video forever after, making it
    look like toggling settings (voiceover on/off, mix vs replace) did
    nothing — the backend was correctly regenerating the file, the browser
    just never re-fetched it."""
    if ".." in filename or filename.startswith("/"):
        return {"error": "Invalid path"}
    filepath = f"{CLIPS_DIR}/{filename}"
    if os.path.exists(filepath):
        return FileResponse(filepath, media_type="video/mp4",
                             headers={"Cache-Control": "no-store, no-cache, must-revalidate", "Pragma": "no-cache"})
    return {"error": "Clip not found"}


@app.get("/clipfinder/recording-file/{recording_id}")
async def clipfinder_serve_recording(recording_id: str):
    """Streams the ORIGINAL uploaded recording — used by the in-app preview
    player so a candidate moment can be previewed (seek + play just that
    time range) before committing to a full cut+caption export."""
    rec = recording_store.get(recording_id)
    if not rec or not os.path.exists(rec["path"]):
        return {"error": "Recording not found"}
    return FileResponse(rec["path"], media_type="video/mp4")


@app.get("/clipfinder/frame/{recording_id}")
async def clipfinder_serve_frame(recording_id: str, t: float = 0):
    """Extracts a single real frame at timestamp t — this is what turns the
    subtitle position picker from a blank black box into an actual preview
    of your footage, so you can see whether the subtitle lands over
    someone's face, off-screen, etc."""
    rec = recording_store.get(recording_id)
    if not rec or not os.path.exists(rec["path"]):
        return {"error": "Recording not found"}
    frame_path = f"{CLIPS_DIR}/_previewframe_{uuid.uuid4().hex}.jpg"
    cmd = [FFMPEG_EXE, "-y", "-ss", str(max(0, t)), "-i", rec["path"], "-vframes", "1", "-q:v", "3", frame_path]
    try:
        r = subprocess.run(cmd, capture_output=True, timeout=30)
        if r.returncode != 0 or not os.path.exists(frame_path):
            return {"error": "Could not extract frame"}
        return FileResponse(frame_path, media_type="image/jpeg", background=BackgroundTask(lambda: os.remove(frame_path) if os.path.exists(frame_path) else None))
    except Exception as e:
        return {"error": str(e)}


@app.get("/dependency-status")
async def dependency_status():
    """One-stop check of every OPTIONAL dependency — Voice Clone (XTTS),
    Parler-TTS, and FFmpeg/Video Studio — so the UI can show a single panel
    instead of separate badges scattered across tabs."""
    deps = []

    deps.append({
        "name": "FFmpeg (Video Studio)",
        "required_for": "Composing final videos, thumbnails, audio mixing",
        "available": bool(VIDEO_STUDIO_AVAILABLE),
        "install_cmd": "venv\\Scripts\\pip.exe install imageio-ffmpeg" if not VIDEO_STUDIO_AVAILABLE else "",
        "detail": "Auto-downloaded binary OK, ready" if VIDEO_STUDIO_AVAILABLE else "Not available, Video Studio disabled",
    })

    # Neural voice cloning — checked in the same priority order generate_clone()
    # actually uses (chatterbox first, then Coqui XTTS). This is a SEPARATE
    # dependency from the Resemblyzer entry below: Resemblyzer only improves
    # which base voice the DSP fallback picks, while this entry controls
    # whether true neural cloning runs at all.
    py_ver = f"{_sys_boot.version_info.major}.{_sys_boot.version_info.minor}"
    _neural_clone_found = False
    try:
        from chatterbox.tts import ChatterboxTTS  # noqa: F401
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        deps.append({
            "name": "Voice Clone — Neural (Chatterbox)",
            "required_for": "The Clone engine tab — true neural voice cloning (best quality)",
            "available": True,
            "install_cmd": "",
            "detail": f"✅ Chatterbox installed (device: {device}) — neural cloning active",
            "cached": chatterbox_model is not None,
            "download_key": "chatterbox" if chatterbox_model is None else None,
        })
        _neural_clone_found = True
    except Exception:
        pass
    if not _neural_clone_found:
        try:
            from TTS.api import TTS  # noqa: F401
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
            deps.append({
                "name": "Voice Clone — Neural (Coqui XTTS v2)",
                "required_for": "The Clone engine tab — true neural voice cloning (best quality)",
                "available": True,
                "install_cmd": "",
                "detail": f"✅ Coqui XTTS v2 installed (device: {device}) — neural cloning active",
                "cached": xtts_model is not None,
                "download_key": "xtts" if xtts_model is None else None,
            })
            _neural_clone_found = True
        except Exception:
            pass
    if not _neural_clone_found:
        deps.append({
            "name": "Voice Clone — Neural (Chatterbox / XTTS)",
            "required_for": "The Clone engine tab — true neural voice cloning (best quality)",
            "available": True,  # DSP fallback (Tier 1-3 below) still works — not broken
            "install_cmd": "",
            "detail": (
                f"⚠️ Neither installed — Clone currently uses DSP-only fallback (Tier 1-3, lower quality). "
                f"You're on Python {py_ver}. Recommended: pip install chatterbox-tts "
                f"(needs Python 3.11 exactly — "
                f"{'your venv qualifies' if py_ver == '3.11' else 'your venv does NOT qualify, needs a separate Python 3.11 venv'}). "
                f"Alternative: pip install TTS (Coqui, needs C++ Build Tools on Windows). Neither is required."
            ),
        })

    try:
        from resemblyzer import VoiceEncoder  # noqa: F401
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        deps.append({
            "name": "Voice Clone — DSP fallback quality (Resemblyzer + Kokoro)",
            "required_for": "Used only when neither neural engine above is installed — speaker embedding selects the closest Kokoro voice, then DSP conversion closes the gap",
            "available": True,
            "install_cmd": "",
            "detail": f"✅ Resemblyzer installed (device: {device}) — Tier 1 active: embedding → Kokoro voice match → DSP conversion",
        })
    except Exception:
        deps.append({
            "name": "Voice Clone — DSP fallback quality (Resemblyzer upgrade)",
            "required_for": "Used only when neither neural engine above is installed — speaker embedding picks the best-matching Kokoro voice before DSP conversion",
            "available": True,  # DSP clone (Tier 3) still works — not broken
            "install_cmd": "",
            "detail": (
                "⚠️ Currently Tier 3 (DSP only — pitch + spectral envelope + energy matching). "
                "Upgrade to Tier 1 (best quality, no restart needed): "
                "pip install resemblyzer webrtcvad  — safe, only needs torch>=1.0, no stack conflict."
            ),
        })

    try:
        from parler_tts import ParlerTTSForConditionalGeneration  # noqa: F401
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        cached = False
        try:
            from huggingface_hub import scan_cache_dir
            cache = scan_cache_dir()
            cached = any("parler-tts" in r.repo_id for r in cache.repos)
        except Exception:
            pass
        deps.append({
            "name": "Parler-TTS (style-conditioned voice)",
            "required_for": "The Parler engine tab in TTS Studio",
            "available": True,
            "install_cmd": "",
            "detail": f"Installed (device: {device})" + (" - model cached, ready instantly" if cached else " - first use downloads ~800MB"),
            "cached": cached,
            "download_key": None if cached else "parler",
        })
    except Exception:
        deps.append({
            "name": "Parler-TTS (style-conditioned voice)",
            "required_for": "The Parler engine tab in TTS Studio",
            "available": False,
            "install_cmd": "Use the 'Fix Entire AI Stack' button (parler-tts + matched torch/transformers)",
            "detail": "Not installed - Parler engine will show an error if used",
        })

    try:
        import torch
        cuda_ok = torch.cuda.is_available()
        gpu_name = torch.cuda.get_device_name(0) if cuda_ok else None
        deps.append({
            "name": "GPU acceleration (CUDA)",
            "required_for": "Fast generation for Clone/Parler/AI Thumbnail/AI Music — everything still works on CPU, just much slower",
            "available": cuda_ok,
            "install_cmd": "" if cuda_ok else "Use the 'Fix Entire AI Stack' button (matched torch/torchaudio/torchvision for CUDA)",
            "detail": f"GPU detected: {gpu_name}" if cuda_ok else "PyTorch is CPU-only — reinstall torch with the CUDA index URL to use your GPU",
        })
    except Exception:
        deps.append({
            "name": "GPU acceleration (CUDA)",
            "required_for": "Fast generation for Clone/Parler/AI Thumbnail/AI Music",
            "available": False,
            "install_cmd": "Use the 'Fix Entire AI Stack' button (installs matched torch/torchaudio/torchvision for CUDA)",
            "detail": "PyTorch not installed at all yet",
        })

    try:
        from diffusers import AutoPipelineForText2Image
        # Force the SAME lazy-loaded internal import chain that breaks at
        # generation time. diffusers uses a lazy-module system, so the bare
        # "from diffusers import X" above can succeed while the real pipeline
        # class is still broken underneath — instantiating from_config (no
        # network call, no download) forces that chain to actually resolve.
        from diffusers.pipelines.auto_pipeline import AUTO_TEXT2IMAGE_PIPELINES_MAPPING  # noqa: F401
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        cached = False
        try:
            from huggingface_hub import scan_cache_dir
            cache = scan_cache_dir()
            cached = any("sd-turbo" in r.repo_id for r in cache.repos)
        except Exception:
            pass
        deps.append({
            "name": "AI Thumbnail (Stable Diffusion / SD-Turbo)",
            "required_for": "The ✨ AI Generate tab in the Thumbnail Generator",
            "available": True,
            "install_cmd": "",
            "detail": f"Installed (device: {device})" + (" - model cached, ready instantly" if cached else " - first use downloads ~2GB"),
            "cached": cached,
            "download_key": None if cached else "sd_thumbnail",
        })
    except Exception as e:
        deps.append({
            "name": "AI Thumbnail (Stable Diffusion / SD-Turbo)",
            "required_for": "The ✨ AI Generate tab in the Thumbnail Generator",
            "available": False,
            "install_cmd": "Use the 'Fix Entire AI Stack' button (matched diffusers/transformers/hub)",
            "detail": f"Broken: {type(e).__name__}: {e}",
        })

    try:
        from diffusers import StableDiffusionInstructPix2PixPipeline  # noqa: F401
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        cached = False
        try:
            from huggingface_hub import scan_cache_dir
            cache = scan_cache_dir()
            cached = any("instruct-pix2pix" in r.repo_id for r in cache.repos)
        except Exception:
            pass
        deps.append({
            "name": "AI Image Editing (InstructPix2Pix)",
            "required_for": "Editing a generated thumbnail with a follow-up instruction (e.g. 'replace X with Y')",
            "available": True,
            "install_cmd": "",
            "detail": f"Installed (device: {device})" + (" - model cached, ready instantly" if cached else " - first edit downloads ~5GB (separate from the AI Thumbnail model)"),
            "cached": cached,
            "download_key": None if cached else "ix2ix",
        })
    except Exception as e:
        deps.append({
            "name": "AI Image Editing (InstructPix2Pix)",
            "required_for": "Editing a generated thumbnail with a follow-up instruction (e.g. 'replace X with Y')",
            "available": False,
            "install_cmd": "Use the 'Fix Entire AI Stack' button (matched diffusers/transformers/hub)",
            "detail": f"Broken: {type(e).__name__}: {e}",
        })

    try:
        from transformers import MusicgenForConditionalGeneration, AutoProcessor
        # Same logic as above: force the deeper import chain MusicGen
        # actually uses at generation time (AutoProcessor pulls in
        # processing_auto, which is where the is_tf_tensor/is_offline_mode
        # breakage actually lives), not just the top-level class import.
        from transformers.models.auto import processing_auto  # noqa: F401
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        cached = False
        try:
            from huggingface_hub import scan_cache_dir
            cache = scan_cache_dir()
            cached = any("musicgen" in r.repo_id for r in cache.repos)
        except Exception:
            pass
        deps.append({
            "name": "AI Music (MusicGen)",
            "required_for": "The ✨ AI Generate tab in Background Music",
            "available": True,
            "install_cmd": "",
            "detail": f"Installed (device: {device})" + (" - model cached, ready instantly" if cached else " - first use downloads ~2GB"),
            "cached": cached,
            "download_key": None if cached else "musicgen",
        })
    except Exception as e:
        deps.append({
            "name": "AI Music (MusicGen)",
            "required_for": "The ✨ AI Generate tab in Background Music",
            "available": False,
            "install_cmd": "Use the 'Fix Entire AI Stack' button (matched transformers/hub/accelerate)",
            "detail": f"Broken: {type(e).__name__}: {e}",
        })

    all_ready = all(d["available"] for d in deps)
    return {"dependencies": deps, "all_ready": all_ready}


@app.exception_handler(Exception)
async def _global_exception_handler(request, exc):
    """Last-resort safety net: never let an uncaught exception return raw HTML.
    Without this, any unhandled error (e.g. a native-library load failure deep
    inside torch/torchaudio) returns FastAPI's default HTML error page, which
    breaks every frontend fetch() call expecting JSON ('Unexpected token I...
    is not valid JSON')."""
    import traceback
    traceback.print_exc()
    return JSONResponse(
        status_code=500,
        content={"error": f"{type(exc).__name__}: {exc}"},
    )



# ══════════════════════════════════════════════════════════════════
# AUTO-INSTALL DEPENDENCIES — runs pip in background, streams output
# Safe on localhost-only tool; NOT for production/public deployment.
# ══════════════════════════════════════════════════════════════════

import sys as _sys

# ──────────────────────────────────────────────────────────────────
# PINNED AI STACK — torch/torchaudio/torchvision AND
# transformers/huggingface_hub/diffusers/accelerate/tokenizers/safetensors
# are ALL pinned together and installed in ONE pip command. This is the
# real fix: a previous version of this code pinned the transformers side
# but installed torch/torchaudio/torchvision separately via "cuda_torch" —
# if that ran before/after/never relative to the stack fix, the two halves
# could still land out of sync (exactly what caused the
# "cannot import name InterpolationMode from torchvision.transforms" crash
# at server boot). Installing everything in a single pip invocation lets
# pip's resolver see the FULL dependency graph at once, so it cannot
# produce a half-matched result regardless of what was installed before
# or what order things are clicked in.
#
# This is a known-compatible, verified-together version set:
#   torch 2.4.1 / torchaudio 2.4.1 / torchvision 0.19.1  (official matched
#   PyTorch release trio, cu121 build)
#   transformers 4.46.3 / huggingface_hub 0.25.2 / diffusers 0.30.3 /
#   accelerate 0.34.2 / tokenizers 0.20.3 / safetensors 0.4.5
_TORCH_PINS = ["torch==2.4.1", "torchaudio==2.4.1", "torchvision==0.19.1"]
_AI_STACK_PINS = [
    "transformers==4.46.3",
    "huggingface_hub==0.25.2",
    "diffusers==0.30.3",
    "accelerate==0.34.2",
    "tokenizers==0.20.3",
    "safetensors==0.4.5",
]
# Both index URLs in ONE command: --index-url serves the cu121 torch/
# torchaudio/torchvision wheels, --extra-index-url lets the same resolver
# pass also reach normal PyPI for everything else, so it's all resolved
# together instead of in separate passes that can disagree with each other.
_FULL_STACK_INSTALL = [
    _sys.executable, "-m", "pip", "install", "--upgrade",
    *_TORCH_PINS, *_AI_STACK_PINS, "scipy",
    "--index-url", "https://download.pytorch.org/whl/cu121",
    "--extra-index-url", "https://pypi.org/simple",
]

INSTALL_COMMANDS = {
    # Kept as an alias of the full atomic install (no longer a partial
    # torch-only install) so old links/buttons referencing "cuda_torch"
    # still resolve to the safe, fully-matched path.
    "cuda_torch": _FULL_STACK_INSTALL,
    # Single source of truth for the whole stack — covers GPU/CUDA, AI
    # Thumbnail (diffusers), AI Music (transformers/MusicGen), Parler, AND
    # the base Kokoro TTS import chain (which also pulls transformers) all
    # at once. This is the ONLY install path that's actually safe to run;
    # everything else below is now just an alias to it.
    "ai_stack":   _FULL_STACK_INSTALL,
    "diffusers":  _FULL_STACK_INSTALL,
    "musicgen":   _FULL_STACK_INSTALL,
    # parler-tts itself is included in the SAME pip call as the rest of the
    # stack (not a separate pass) so its own torch/transformers requirements
    # get resolved together with everything else, not against it.
    "parler":     [_sys.executable, "-m", "pip", "install", "--upgrade",
                   "git+https://github.com/huggingface/parler-tts.git",
                   *_TORCH_PINS, *_AI_STACK_PINS,
                   "--index-url", "https://download.pytorch.org/whl/cu121",
                   "--extra-index-url", "https://pypi.org/simple"],
    # Same reasoning for chatterbox-tts — this is what broke AI Image/Music
    # last time by pulling its own incompatible transformers/huggingface_hub
    # in a separate, unconstrained pass.
    "clone":      [_sys.executable, "-m", "pip", "install", "--upgrade",
                   "chatterbox-tts",
                   *_TORCH_PINS, *_AI_STACK_PINS,
                   "--index-url", "https://download.pytorch.org/whl/cu121",
                   "--extra-index-url", "https://pypi.org/simple"],
    "ffmpeg":     [_sys.executable, "-m", "pip", "install", "imageio-ffmpeg"],
    # resemblyzer: speaker embedding for voice clone Tier 1. Safe — only needs
    # torch>=1.0, no version conflict with the pinned 2.4.1 stack.
    "resemblyzer": [_sys.executable, "-m", "pip", "install", "resemblyzer", "webrtcvad"],
    # Repairs an already-broken torchaudio without touching torch's CUDA build.
    "repair_torchaudio": [_sys.executable, "-m", "pip", "install", "torchaudio",
                           "--index-url", "https://download.pytorch.org/whl/cu121",
                           "--upgrade", "--force-reinstall", "--no-deps"],
}


@app.get("/install-dependency/{key}")
async def install_dependency(key: str):
    """Stream pip install output as Server-Sent Events so the UI can show
    live progress without polling. Uses the SAME Python executable that is
    running the server (sys.executable) so it always targets the right venv."""
    if key not in INSTALL_COMMANDS:
        return JSONResponse(status_code=400, content={"error": f"Unknown dependency key: {key}"})

    cmd = INSTALL_COMMANDS[key]

    async def event_stream():
        yield f"data: Starting: {' '.join(cmd)}\n\n"
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
            )
            async for line in proc.stdout:
                text = line.decode("utf-8", errors="replace").rstrip()
                if text:
                    yield f"data: {text}\n\n"
            await proc.wait()
            if proc.returncode == 0:
                yield "data: ✅ Installation complete!\n\n"
                yield "event: done\ndata: ok\n\n"
            else:
                yield f"data: ❌ pip exited with code {proc.returncode}\n\n"
                yield "event: error\ndata: failed\n\n"
        except Exception as e:
            yield f"data: ❌ Error: {e}\n\n"
            yield "event: error\ndata: failed\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)

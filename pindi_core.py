import atexit
import json
import os
import re
import shutil
import subprocess
import tempfile
import threading
import time
from datetime import datetime

# Coqui XTTS-v2 license (non-commercial) accept - needed for model download
os.environ["COQUI_TOS_AGREED"] = "1"

import sys
from pathlib import Path

# In a windowed .exe, stdout/stderr are None, which crashes some libraries
if sys.stdout is None:
    sys.stdout = open(os.devnull, "w")
if sys.stderr is None:
    sys.stderr = open(os.devnull, "w")

# If a "models" folder sits next to the app, load models from there (offline use)
BASE_DIR = Path(sys.executable).parent if getattr(sys, "frozen", False) else Path(__file__).parent
if (BASE_DIR / "models").exists():
    os.environ["TTS_HOME"] = str(BASE_DIR / "models")
    # Hugging Face / PyTorch downloads that were packed with the app are also used from there
    if (BASE_DIR / "models" / "hf").exists():
        os.environ["HF_HOME"] = str(BASE_DIR / "models" / "hf")
    if (BASE_DIR / "models" / "torch").exists():
        os.environ["TORCH_HOME"] = str(BASE_DIR / "models" / "torch")

import gradio as gr
import torch
from TTS.api import TTS

import pindi_chat

chat_server = pindi_chat.ChatServer(Path.home() / ".zynth_studio" / "chat.db")
atexit.register(pindi_chat.stop_tunnel)


def _patch_torchaudio():
    """Newer torchaudio reads/writes audio through torchcodec, which needs FFmpeg DLLs that
    many PCs do not have (this caused the 'libtorchcodec_core' errors). All audio the app
    uses is WAV/FLAC, which soundfile reads directly, so use that and keep torchaudio's own
    loader only as a fallback."""
    try:
        import soundfile as sf
        import torchaudio
    except Exception:
        return
    orig_load, orig_save = torchaudio.load, torchaudio.save
    simple_load = {"frame_offset", "num_frames", "normalize", "channels_first"}

    def load(uri, *args, **kwargs):
        if not args and set(kwargs) <= simple_load:
            try:
                src = uri if hasattr(uri, "read") else str(uri)
                data, rate = sf.read(
                    src, start=kwargs.get("frame_offset", 0), frames=kwargs.get("num_frames", -1),
                    dtype="float32", always_2d=True,
                )
                tensor = torch.from_numpy(data)
                if kwargs.get("channels_first", True):
                    tensor = tensor.T.contiguous()
                return tensor, rate
            except Exception:
                pass
        return orig_load(uri, *args, **kwargs)

    def save(uri, src, sample_rate, *args, **kwargs):
        name = str(uri).lower()
        if not args and set(kwargs) <= {"channels_first"} and name.endswith((".wav", ".flac", ".ogg")):
            try:
                data = src.detach().cpu().float()
                if kwargs.get("channels_first", True):
                    data = data.T
                sf.write(str(uri), data.numpy(), int(sample_rate))
                return None
            except Exception:
                pass
        return orig_save(uri, src, sample_rate, *args, **kwargs)

    torchaudio.load = load
    torchaudio.save = save


_patch_torchaudio()

APP_NAME = "Pindi Studio"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ---------- Settings (remembered folders) ----------
SETTINGS_FILE = Path.home() / ".zynth_studio_settings.json"
DOCS_DIR = Path.home() / "Documents" / "Pindi Studio"
try:  # the folder used to be called "Zynth Studio": keep everything (saved voices, downloads) by renaming it
    _OLD_DOCS = Path.home() / "Documents" / "Zynth Studio"
    if _OLD_DOCS.exists() and not DOCS_DIR.exists():
        _OLD_DOCS.rename(DOCS_DIR)
except OSError:
    pass
DEFAULT_SAVE_DIR = str(DOCS_DIR)
DEFAULT_DOWNLOAD_DIR = str(DOCS_DIR / "Downloads")
VOICES_DIR = DOCS_DIR / "Saved Voices"


def load_setting(key, default):
    try:
        data = json.loads(SETTINGS_FILE.read_text(encoding="utf-8"))
        value = data.get(key)
        if value:
            return value
    except Exception:
        pass
    return default


def store_setting(key, value):
    try:
        try:
            data = json.loads(SETTINGS_FILE.read_text(encoding="utf-8"))
        except Exception:
            data = {}
        data[key] = value
        SETTINGS_FILE.write_text(json.dumps(data), encoding="utf-8")
    except Exception:
        pass


# ---------- Audio input helpers (accept any audio/video file) ----------
def get_ffmpeg():
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return shutil.which("ffmpeg")


def pick_path(value):
    """Return a plain file path from whatever Gradio hands us."""
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        value = value[0] if value else None
    if value is None:
        return None
    if isinstance(value, dict):
        return value.get("path") or value.get("name")
    return value if isinstance(value, str) else getattr(value, "name", None)


def to_wav(path):
    """Convert any audio/video file to a clean mono WAV the models can read."""
    exe = get_ffmpeg()
    if not exe:
        raise gr.Error("ffmpeg was not found. Close the app and run run.bat again to install it.")
    out = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
    cmd = [exe, "-y", "-i", path, "-vn", "-ac", "1", "-ar", "24000", "-c:a", "pcm_s16le", out]
    flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    try:
        result = subprocess.run(cmd, capture_output=True, creationflags=flags)
    except OSError as e:
        raise gr.Error(f"Could not run ffmpeg: {e}")
    if result.returncode != 0 or not os.path.exists(out) or os.path.getsize(out) < 1000:
        raise gr.Error("Could not read this file. Please check that it contains audio and is not damaged.")
    return out


def get_input(file_value, mic_value, what):
    path = pick_path(file_value) or pick_path(mic_value)
    if not path:
        raise gr.Error(f"Please add {what}: upload any audio or video file, or record from the microphone.")
    return to_wav(path)


# ---------- Saved voices ----------
NO_SAVED = "None - use a new sample"


def clean_name(name):
    name = re.sub(r'[\\/:*?"<>|]+', "", (name or "")).strip().strip(".")
    return name[:60]


def list_voices():
    try:
        return sorted(p.stem for p in VOICES_DIR.glob("*.wav"))
    except OSError:
        return []


def voice_choices():
    return [NO_SAVED] + list_voices()


def saved_voice_path(choice):
    if not choice or choice == NO_SAVED:
        return None
    path = VOICES_DIR / f"{choice}.wav"
    return str(path) if path.exists() else None


def get_voice_input(file_value, mic_value, saved_choice, what):
    """A freshly added file/recording wins; otherwise use the selected saved voice."""
    path = pick_path(file_value) or pick_path(mic_value)
    if path:
        return to_wav(path)
    saved = saved_voice_path(saved_choice)
    if saved:
        return saved
    raise gr.Error(
        f"Please add {what}: pick a saved voice, upload any audio or video file, or record from the microphone."
    )


CLEAN_FILTER = (
    "highpass=f=70,afftdn=nr=18:nf=-30:tn=1,"
    "silenceremove=start_periods=1:start_threshold=-50dB:stop_periods=-1:stop_duration=0.7:stop_threshold=-50dB,"
    "loudnorm=I=-18:TP=-2:LRA=11"
)
POLISH_FILTER = "highpass=f=60,afftdn=nr=12:nf=-35"
SOURCE_CLEAN_FILTER = "highpass=f=70,afftdn=nr=15:nf=-30:tn=1"  # keeps the timing: no silence cutting
_clean_cache = {}


def run_filter(src, filt, rate="24000"):
    """Run an ffmpeg audio filter. Returns the new file, or the original if anything fails."""
    exe = get_ffmpeg()
    if not exe:
        return src
    out = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
    cmd = [exe, "-y", "-i", src, "-af", filt] + (["-ar", rate] if rate else []) + ["-ac", "1", out]
    try:
        r = subprocess.run(cmd, capture_output=True, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        if r.returncode == 0 and os.path.exists(out) and os.path.getsize(out) > 48000 * 2:
            return out
    except Exception:
        pass
    return src


def clean_sample(path):
    """Remove background hiss/hum from a voice sample, level its volume and cut long silences.
    The cloned voice copies the sample's noise, so a cleaner sample gives a cleaner voice."""
    try:
        st = os.stat(path)
        key = (path, st.st_mtime_ns, st.st_size)
    except OSError:
        return path
    if key not in _clean_cache or not os.path.exists(_clean_cache[key]):
        if len(_clean_cache) > 8:
            _clean_cache.clear()
        _clean_cache[key] = run_filter(path, CLEAN_FILTER)
    return _clean_cache[key]


def save_voice(name, file_value, mic_value):
    safe = clean_name(name)
    if not safe:
        raise gr.Error("Please type a name for this voice first.")
    path = pick_path(file_value) or pick_path(mic_value)
    if not path:
        raise gr.Error("Please add a voice sample (upload or record) before saving it.")
    wav = to_wav(path)
    try:
        VOICES_DIR.mkdir(parents=True, exist_ok=True)
        shutil.copy(wav, VOICES_DIR / f"{safe}.wav")
    except OSError as e:
        raise gr.Error(f"Could not save the voice: {e}")
    gr.Info(f'Voice "{safe}" saved. You can pick it from the "Saved voice" list now.')
    update = gr.update(choices=voice_choices(), value=safe)
    return update, update


def delete_voice(choice):
    if not choice or choice == NO_SAVED:
        raise gr.Error('Please select a saved voice to delete first.')
    path = VOICES_DIR / f"{choice}.wav"
    try:
        if path.exists():
            path.unlink()
    except OSError as e:
        raise gr.Error(f"Could not delete the voice: {e}")
    gr.Info(f'Voice "{choice}" deleted.')
    update = gr.update(choices=voice_choices(), value=NO_SAVED)
    return update, update


def preview_voice(choice):
    return saved_voice_path(choice)


# ---------- Save folders ----------
def save_result(tmp_path, prefix, folder, label=None, ext=".wav", exact_name=None):
    """Copy the result into the chosen folder and return the final path.
    exact_name: the file is named exactly like that (a number is added only if the name is already taken)."""
    folder = (folder or "").strip().strip('"')
    if not folder:
        raise gr.Error("Please type a save folder path or choose one with Browse.")
    try:
        Path(folder).mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        name = f"{prefix}_{clean_name(label)}_{stamp}" if label else f"{prefix}_{stamp}"
        if exact_name and clean_name(exact_name):
            name = clean_name(exact_name)
        final = Path(folder) / f"{name}{ext}"
        n = 2
        while final.exists():  # never overwrite an earlier result
            final = Path(folder) / f"{name}_{n}{ext}"
            n += 1
        shutil.copy(tmp_path, final)
    except OSError as e:
        raise gr.Error(f"Could not save to this folder: {e}")
    store_setting("save_dir", folder)
    return str(final)


def main_window():
    import webview

    for w in webview.windows:
        if w.title == APP_NAME:
            return w
    return webview.windows[-1] if webview.windows else None


def browse_folder(current):
    try:
        import webview

        win = main_window()
        if win is None:
            raise gr.Error("Browse only works in the desktop window. Please type the path instead.")
        dialog_type = webview.FileDialog.FOLDER if hasattr(webview, "FileDialog") else webview.FOLDER_DIALOG
        start = current if current and Path(current).exists() else ""
        result = win.create_file_dialog(dialog_type, directory=start)
        if result:
            return result[0] if isinstance(result, (list, tuple)) else result
    except gr.Error:
        raise
    except Exception as e:
        raise gr.Error(f"Could not open the folder picker: {e}. Please type the path instead.")
    return current


def open_folder(folder):
    folder = (folder or "").strip().strip('"')
    if not folder:
        raise gr.Error("Please enter a folder path first.")
    Path(folder).mkdir(parents=True, exist_ok=True)
    try:
        os.startfile(folder)  # Windows only
    except AttributeError:
        raise gr.Error("Opening the folder is only supported on Windows.")
    return None


# ---------- Models ----------
_tts = None
_fast_tts = None
_load_lock = threading.Lock()


def get_tts():
    global _tts
    with _load_lock:
        if _tts is None:
            print("Loading XTTS-v2 model (first run downloads ~2 GB)...")
            _tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(DEVICE)
    return _tts


POCKET_MODEL = "Fast - Pocket TTS (CPU real-time, English only)"
_pocket = None
_pocket_voices = {}


def get_pocket():
    """Kyutai Pocket TTS: a small voice-cloning model made to run in real time on a normal CPU."""
    global _pocket
    with _load_lock:
        if _pocket is None:
            try:
                from pocket_tts import TTSModel
            except ImportError:
                raise gr.Error(
                    "The fast model is not installed yet. Close the app and run run.bat again "
                    "(it installs it), or type:  venv\\Scripts\\pip install pocket-tts"
                )
            print("Loading Pocket TTS (first run downloads the model)...")
            _pocket = TTSModel.load_model()
    return _pocket


def pocket_voice(model, sample):
    st = os.stat(sample)
    key = (sample, st.st_mtime_ns, st.st_size)
    if key not in _pocket_voices:
        if len(_pocket_voices) > 8:
            _pocket_voices.clear()
        try:
            _pocket_voices[key] = model.get_state_for_audio_prompt(sample)
        except ValueError as e:
            if "cloning" in str(e).lower():
                raise gr.Error(
                    "Pocket TTS needs a free Hugging Face login to copy voices:\n"
                    "1) Sign up at huggingface.co and click 'Agree' on https://huggingface.co/kyutai/pocket-tts\n"
                    "2) Create a token (Settings > Access Tokens, type Read)\n"
                    "3) Close the app, open cmd in the app folder and run:  venv\\Scripts\\hf auth login\n"
                    "Then start the app again. Until then, use the XTTS-v2 model."
                )
            raise
    return _pocket_voices[key]


_light_tts = None
BASE_SPEAKER = "Claribel Dervla"  # one of XTTS-v2's built-in voices, used only as the plain base voice


def get_light_tts():
    """A small English text-to-speech model (VITS, ~150 MB). It only makes the plain base speech;
    the voice changer then turns it into the sample's voice."""
    global _light_tts
    with _load_lock:
        if _light_tts is None:
            print("Loading the light English voice (first run downloads ~150 MB)...")
            _light_tts = TTS("tts_models/en/ljspeech/vits").to(DEVICE)
    return _light_tts


def make_base_part(language, chunk, part):
    code = LANGUAGES[language]
    if code == "en":
        get_light_tts().tts_to_file(text=chunk, file_path=part)
    else:  # the light model is English only, so other languages use XTTS's built-in voice as the base
        get_tts().tts_to_file(text=chunk, speaker=BASE_SPEAKER, language=code, file_path=part)


def convert_voice(model_choice, source, target, steps):
    """Turn `source` speech into the voice of `target`, with the same engines as the Voice Changer tab."""
    if model_choice == SEEDVC_LABEL:
        return run_seedvc(source, target, steps)
    out = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
    get_vc(model_choice).voice_conversion_to_file(source_wav=source, target_wav=target, file_path=out)
    return out


def get_fast_tts():
    """YourTTS: a much lighter voice-cloning model that runs in about real time on a CPU."""
    global _fast_tts
    with _load_lock:
        if _fast_tts is None:
            print("Loading YourTTS model (first run downloads ~400 MB)...")
            _fast_tts = TTS("tts_models/multilingual/multi-dataset/your_tts").to(DEVICE)
    return _fast_tts


# ---------- Seed-VC (runs in its own Python 3.10 environment, set up by setup_seedvc.bat) ----------
SEEDVC_DIR = BASE_DIR / "seedvc"
SEEDVC_REPO = SEEDVC_DIR / "repo"
# A portable Python (seedvc/python, made by build_exe_light_seedvc.bat) is used when present,
# otherwise the environment created by setup_seedvc.bat (seedvc/env).
_SV_PORTABLE = SEEDVC_DIR / "python" / ("python.exe" if os.name == "nt" else "bin/python")
_SV_ENV = SEEDVC_DIR / "env" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
SEEDVC_PY = _SV_PORTABLE if _SV_PORTABLE.exists() else _SV_ENV
SEEDVC_LABEL = "Seed-VC (highest accuracy)"


def seedvc_ready():
    return SEEDVC_PY.exists() and (SEEDVC_REPO / "inference.py").exists()


# Seed-VC runs in a background worker that keeps its models loaded, so only the first
# conversion pays the (slow) model loading time. Later conversions start immediately.
SEEDVC_WORKER_CODE = r"""
import json, sys, traceback
from argparse import Namespace

_real_stdout = sys.stdout
sys.stdout = sys.stderr  # prints from the model code must not corrupt our replies

import inference

_orig_load_models = inference.load_models
_loaded = {}


def _cached_load_models(args):
    key = (args.checkpoint, args.config, args.f0_condition, args.fp16)
    if key not in _loaded:
        _loaded[key] = _orig_load_models(args)
    return _loaded[key]


inference.load_models = _cached_load_models


def reply(obj):
    _real_stdout.write(json.dumps(obj) + "\n")
    _real_stdout.flush()


reply({"ready": True})
while True:
    line = sys.stdin.readline()
    if not line:  # the app closed: stop
        break
    line = line.strip()
    if not line:
        continue
    try:
        req = json.loads(line)
        args = Namespace(
            source=req["source"], target=req["target"], output=req["output"],
            diffusion_steps=int(req["steps"]), length_adjust=1.0, inference_cfg_rate=0.7,
            f0_condition=False, auto_f0_adjust=False, semi_tone_shift=0,
            checkpoint=None, config=None, fp16=False,
        )
        inference.main(args)
        reply({"ok": True})
    except BaseException:
        reply({"ok": False, "error": traceback.format_exc()[-1500:]})
"""

_sv_proc = None
_sv_lock = threading.Lock()
_SV_LOG = Path(tempfile.gettempdir()) / "zynth_seedvc_worker.log"


def _sv_log_tail(n=900):
    try:
        return _SV_LOG.read_text(encoding="utf-8", errors="replace").strip()[-n:]
    except OSError:
        return ""


def _sv_stop():
    global _sv_proc
    proc, _sv_proc = _sv_proc, None
    if proc is not None and proc.poll() is None:
        try:
            proc.stdin.close()
            proc.terminate()
        except Exception:
            pass


atexit.register(_sv_stop)


def _sv_start():
    global _sv_proc
    worker = SEEDVC_REPO / "zynth_worker.py"
    try:
        if not worker.exists() or worker.read_text(encoding="utf-8") != SEEDVC_WORKER_CODE:
            worker.write_text(SEEDVC_WORKER_CODE, encoding="utf-8")
        log = open(_SV_LOG, "w", encoding="utf-8")
    except OSError as e:
        raise gr.Error(f"Could not prepare Seed-VC: {e}")
    env = dict(os.environ, PYTHONUTF8="1", PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1")
    for var in ("PYTHONHOME", "PYTHONPATH"):  # keep the main app's Python settings out of Seed-VC's Python
        env.pop(var, None)
    flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    print("Starting Seed-VC (first conversion loads its models; later ones are faster)...")
    try:
        proc = subprocess.Popen(
            [str(SEEDVC_PY), "zynth_worker.py"], cwd=str(SEEDVC_REPO), env=env,
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=log,
            text=True, encoding="utf-8", bufsize=1, creationflags=flags,
        )
    except OSError as e:
        raise gr.Error(f"Could not start Seed-VC: {e}")
    first = proc.stdout.readline()
    if not first:
        proc.wait()
        raise gr.Error(f"Seed-VC could not start:\n{_sv_log_tail()}")
    _sv_proc = proc


def run_seedvc(source, target, steps):
    global _sv_proc
    if not seedvc_ready():
        raise gr.Error(
            "Seed-VC is not installed yet. Close the app, run setup_seedvc.bat once, "
            "then start the app again."
        )
    with _sv_lock:
        if _sv_proc is None or _sv_proc.poll() is not None:
            _sv_start()
        out_dir = tempfile.mkdtemp(prefix="seedvc_")
        request = {"source": source, "target": target, "output": out_dir, "steps": int(steps)}
        try:
            _sv_proc.stdin.write(json.dumps(request) + "\n")
            _sv_proc.stdin.flush()
            line = _sv_proc.stdout.readline()
        except (OSError, ValueError):
            line = ""
        if not line:
            _sv_proc = None
            raise gr.Error(f"Seed-VC stopped unexpectedly (it may have run out of memory):\n{_sv_log_tail()}")
        try:
            reply = json.loads(line)
        except ValueError:
            raise gr.Error(f"Seed-VC gave an unreadable answer:\n{line[:300]}")
    if not reply.get("ok"):
        raise gr.Error(f"Seed-VC failed:\n{reply.get('error', '')}")
    wavs = sorted(Path(out_dir).glob("*.wav"), key=lambda p: p.stat().st_mtime)
    if not wavs:
        raise gr.Error("Seed-VC finished but no output file was created.")
    return str(wavs[-1])


VC_MODELS = {
    "OpenVoice v2 (best match, recommended)": "voice_conversion_models/multilingual/multi-dataset/openvoice_v2",
    "OpenVoice v1": "voice_conversion_models/multilingual/multi-dataset/openvoice_v1",
    "kNN-VC (basic; works better with a longer sample)": "voice_conversion_models/multilingual/multi-dataset/knnvc",
    "FreeVC24 (fast, basic match)": "voice_conversion_models/multilingual/vctk/freevc24",
}
VC_CHOICES = [SEEDVC_LABEL] + list(VC_MODELS)
VC_DEFAULT = SEEDVC_LABEL if seedvc_ready() else list(VC_MODELS)[0]
_vc_cache = {}


def get_vc(choice):
    if choice not in _vc_cache:
        name = VC_MODELS[choice]
        print(f"Loading voice changer model: {name} (first run downloads it)...")
        try:
            _vc_cache[choice] = TTS(name).to(DEVICE)
        except Exception as e:
            raise gr.Error(
                f"Could not load '{choice}': {e}. Try another model from the list, "
                "or update with: pip install -U coqui-tts"
            )
    return _vc_cache[choice]


LANGUAGES = {
    "English": "en",
    "Hindi (try for Urdu)": "hi",
    "Arabic": "ar",
    "Spanish": "es",
    "French": "fr",
    "German": "de",
    "Italian": "it",
    "Portuguese": "pt",
    "Polish": "pl",
    "Turkish": "tr",
    "Russian": "ru",
    "Dutch": "nl",
    "Czech": "cs",
    "Chinese": "zh-cn",
    "Japanese": "ja",
    "Korean": "ko",
    "Hungarian": "hu",
}

MAX_TTS_CHARS = 5000      # change this number to allow longer text
TTS_CHUNK = 230           # XTTS works best with pieces under ~250 characters


def split_text(text, limit=TTS_CHUNK):
    """Cut long text into short pieces at sentence ends, so the voice engine handles each well."""
    text = re.sub(r"[ \t]+", " ", text.replace("\r", "")).strip()
    sentences = []
    for para in re.split(r"\n+", text):
        para = para.strip()
        if not para:
            continue
        sentences += [x.strip() for x in re.findall(r".+?(?:[.!?\u06d4\u0964\u3002]+[\"')\]]*(?:\s+|$)|$)", para) if x.strip()]
    pieces = []
    for sen in sentences:
        while len(sen) > limit:  # a very long sentence: cut at a comma or space
            cut = max(sen.rfind(c, 0, limit) for c in (",", ";", ":", "\u060c"))
            if cut < limit // 3:
                cut = sen.rfind(" ", 0, limit)
            if cut < 1:
                cut = limit
            pieces.append(sen[: cut + 1].strip())
            sen = sen[cut + 1:].strip()
        if sen:
            pieces.append(sen)
    chunks, cur = [], ""
    for piece in pieces:  # join short sentences together
        if cur and len(cur) + 1 + len(piece) > limit:
            chunks.append(cur)
            cur = piece
        else:
            cur = f"{cur} {piece}".strip()
    if cur:
        chunks.append(cur)
    return chunks


def join_wavs(items, out_path, gap_seconds=0.18):
    """Join WAV files. A number in `items` is a pause of that many seconds; between two parts
    with no pause in between, a short natural gap is used."""
    import wave

    first = next(i for i in items if isinstance(i, str))
    with wave.open(first, "rb") as f:
        params = f.getparams()
    frame = params.sampwidth * params.nchannels

    def silence(sec):
        return b"\x00" * (int(params.framerate * sec) * frame)

    with wave.open(out_path, "wb") as out:
        out.setparams(params)
        prev_part = False
        pending = None  # a requested pause not written yet
        for item in items:
            if not isinstance(item, str):
                pending = (pending or 0.0) + float(item)
                continue
            if pending is not None:
                out.writeframes(silence(pending))
            elif prev_part:
                out.writeframes(silence(gap_seconds))
            pending = None
            with wave.open(item, "rb") as part:
                out.writeframes(part.readframes(part.getnframes()))
            prev_part = True
        if pending:
            out.writeframes(silence(pending))


BREAK_RE = re.compile(r"[<\[]\s*break\b([^>\]]*?)/?\s*[>\]]", re.I)
MAX_BREAK_SECONDS = 30.0


def break_seconds(attrs):
    """Length of a <break time="1500ms" /> tag in seconds (1500ms = 1.5s, "2s" = 2 seconds). A bare tag = 0.5 s."""
    t = re.search(r"time\s*[=:]?\s*[\"'\u201c\u201d\u2018\u2019]?\s*([0-9]*\.?[0-9]+)\s*(ms|milliseconds?|s|sec|seconds?)?", attrs or "", re.I)
    if t:
        sec = float(t.group(1))
        if (t.group(2) or "s").lower().startswith("m"):
            sec /= 1000.0
    else:
        sec = 0.5
    return min(sec, MAX_BREAK_SECONDS)


def parse_script(text):
    """Split the script into spoken pieces and pauses. A tag like <break time="900ms" /> (or
    time="0.9s") makes the voice stop for that long. Returns [("text", str) | ("pause", seconds)]."""
    items = []
    pos = 0
    for m in BREAK_RE.finditer(text):
        for chunk in split_text(text[pos:m.start()]):
            items.append(("text", chunk))
        items.append(("pause", break_seconds(m.group(1))))
        pos = m.end()
    for chunk in split_text(text[pos:]):
        items.append(("text", chunk))
    return items


BEST_MODEL = "Best quality - XTTS-v2"
FAST_MODEL = "Basic - YourTTS (English / French / Portuguese)"
MATCH_MODEL = "Voice Changer match - light TTS + voice changer (same match as the Voice Changer tab)"
MAX_MODEL = "Max match - XTTS-v2 + Seed-VC (most accurate, slowest)"
INWORLD_MODEL = "Inworld 360 (online, needs INWORLD_API_KEY)"
def pocket_available():
    """Pocket TTS is only offered once it is installed AND a Hugging Face login exists."""
    try:
        import importlib.util

        import huggingface_hub

        return importlib.util.find_spec("pocket_tts") is not None and bool(huggingface_hub.get_token())
    except Exception:
        return False


TTS_MODELS = [MATCH_MODEL, BEST_MODEL] + ([POCKET_MODEL] if pocket_available() else []) + [FAST_MODEL, MAX_MODEL, INWORLD_MODEL]
FAST_LANGS = {"en": "en", "fr": "fr-fr", "pt": "pt-br"}

_latent_cache = {}


def xtts_latents(model, sample):
    """The voice 'fingerprint' of a sample. Worked out once and reused for every part of the text
    (and for later generations with the same saved voice) instead of once per sentence."""
    st = os.stat(sample)
    key = (sample, st.st_mtime_ns, st.st_size)
    if key not in _latent_cache:
        if len(_latent_cache) > 8:
            _latent_cache.clear()
        # use up to 30 seconds of the sample (the default only looks at the first 10 seconds)
        _latent_cache[key] = model.get_conditioning_latents(
            audio_path=[sample], gpt_cond_len=30, gpt_cond_chunk_len=4, max_ref_length=60
        )
    return _latent_cache[key]


def change_tempo(path, speed):
    """Speed up / slow down a finished WAV file (used when the model cannot do it itself)."""
    if abs(speed - 1.0) < 0.01:
        return
    tmp = path + ".tempo.wav"
    try:
        cmd = [get_ffmpeg(), "-y", "-i", path, "-filter:a", f"atempo={speed:.3f}", tmp]
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        r = subprocess.run(cmd, capture_output=True, creationflags=flags)
        if r.returncode == 0 and os.path.exists(tmp) and os.path.getsize(tmp) > 1000:
            os.replace(tmp, path)
    except Exception:
        pass
    finally:
        try:
            if os.path.exists(tmp):
                os.remove(tmp)
        except OSError:
            pass


def make_part_xtts(tts, state, sample, chunk, language, speed, part):
    """One piece of text with XTTS. Uses the cached voice fingerprint; if this XTTS version
    does not support that, quietly falls back to the standard (slower) call."""
    if state.get("direct", True):
        try:
            import numpy as np
            import soundfile as sf

            model = tts.synthesizer.tts_model
            lat, emb = xtts_latents(model, sample)
            try:
                out = model.inference(chunk, LANGUAGES[language], lat, emb, temperature=0.6, speed=float(speed))
            except TypeError:
                out = model.inference(chunk, LANGUAGES[language], lat, emb, temperature=0.6)
                state["stretch"] = True
            rate = int(getattr(getattr(model.config, "audio", None), "output_sample_rate", 24000))
            sf.write(part, np.asarray(out["wav"], dtype="float32"), rate)
            return
        except Exception as e:
            print(f"Fast XTTS path not available ({e}); using the standard path.")
            state["direct"] = False
    tts.tts_to_file(
        text=chunk, speaker_wav=sample, language=LANGUAGES[language], file_path=part, speed=float(speed)
    )


# ---------- Inworld 360 (online text-to-speech through the Inworld AI HTTP streaming API) ----------
# The API key is read ONLY from the INWORLD_API_KEY environment variable of this computer. It is never
# stored in the code, never shown in the page and never written to a log or an error message.
import base64
import http.client
import urllib.error
import urllib.request

INWORLD_URL = "https://api.inworld.ai/tts/v1/voice:stream"
INWORLD_KEY_VAR = "INWORLD_API_KEY"
INWORLD_MODEL_ID = "inworld-tts-2"
INWORLD_VOICE = "Timothy"        # default voice
# Voices shown in the dropdown. Any other exact voice name from your Inworld Portal can also be typed in.
INWORLD_VOICES = ["Timothy", "Blake", "Sarah", "Jason", "Nate", "Luna", "Hades", "Simon", "Reed", "Tyler",
                  "Olivia", "Hana", "Ashley", "Aarav", "Mark"]


def inworld_voice_list():
    """The voices in the dropdown: the saved list (so added voices stay), else the built-in list."""
    saved = load_setting("inworld_voices", None)
    if isinstance(saved, list):
        names = [str(x).strip() for x in saved if str(x).strip()]
        if names:
            return names
    return list(INWORLD_VOICES)


def inworld_add_voice(name):
    """Add a voice name (exactly as written in the Inworld Portal) to the dropdown and select it."""
    name = re.sub(r"[\x00-\x1f]", "", name or "").strip()[:200]
    if not name:
        raise gr.Error("Type the exact voice name from Inworld first.")
    voices = inworld_voice_list()
    found = next((v for v in voices if v.lower() == name.lower()), None)
    if found is None:
        voices.append(name)
        store_setting("inworld_voices", voices)
        found = name
        gr.Info(f"Voice '{name}' added.")
    else:
        gr.Info(f"'{found}' is already in the list.")
    store_setting("inworld_voice", found)
    return gr.update(choices=voices, value=found), ""


def inworld_remove_voice(current):
    """Take the selected voice out of the dropdown (it can be added again any time)."""
    voices = inworld_voice_list()
    found = next((v for v in voices if v.lower() == (current or "").strip().lower()), None)
    if found is None:
        raise gr.Error("This voice is not in the list.")
    if len(voices) <= 1:
        raise gr.Error("At least one voice has to stay in the list.")
    voices.remove(found)
    store_setting("inworld_voices", voices)
    store_setting("inworld_voice", voices[0])
    gr.Info(f"Voice '{found}' removed from the list.")
    return gr.update(choices=voices, value=voices[0])


def inworld_voice_id(value=None):
    """The voice to use: the one chosen in the app, else the last one used, else the default."""
    voice = (value or "").strip() or str(load_setting("inworld_voice", "") or "").strip() or INWORLD_VOICE
    return voice[:200]
INWORLD_DELIVERY = "BALANCED"
INWORLD_LANGUAGE = "AUTO"        # AUTO = Inworld detects the language from the text (the field is simply left out)
INWORLD_MAX_UNITS = 4000         # Inworld accepts at most 4,000 UTF-16 code units per streaming request
INWORLD_TIMEOUT = 90             # seconds to wait for the server on each read


class InworldError(Exception):
    """A readable Inworld problem (shown to the user as it is)."""


def _utf16_len(text):
    return len(text.encode("utf-16-le")) // 2


def inworld_key():
    key = (os.environ.get(INWORLD_KEY_VAR) or "").strip()
    if key.lower().startswith("basic "):  # also accept the value with the word Basic already in front
        key = key[6:].strip()
    if not key:
        raise gr.Error(
            f"Inworld 360 is not set up on this computer: the {INWORLD_KEY_VAR} environment variable is missing. "
            f"Add it in Windows (Start menu > 'Edit the system environment variables' > Environment Variables > New), "
            f"then close and start Pindi Studio again."
        )
    return key


def _inworld_api_message(body):
    """Pick the human message out of an Inworld error body (never anything else from the request)."""
    try:
        data = json.loads(body.decode("utf-8", "replace"))
        data = data.get("error", data) if isinstance(data, dict) else {}
        return str(data.get("message") or "").strip()
    except Exception:
        return ""


def _inworld_http_error(status, body):
    detail = _inworld_api_message(body)
    tail = f" ({detail})" if detail else ""
    if status in (401, 403):
        return ("Inworld rejected the API key (invalid authentication). Check the " + INWORLD_KEY_VAR +
                " value: use the Base64 credentials from the Inworld Portal exactly as shown, without changing them." + tail)
    if status == 429:
        return "Inworld says there were too many requests or the quota is used up. Wait a moment and try again." + tail
    if status == 404:
        return ("Inworld could not find this voice. Check that the voice name is written exactly as in your "
                "Inworld Portal voice list." + tail)
    if status == 400:
        return "Inworld could not use this request" + (f": {detail}" if detail else ".")
    if status >= 500:
        return f"Inworld is not available right now (server error {status}). Please try again in a little while." + tail
    return f"Inworld returned an error (HTTP {status})." + tail


def inworld_stream_to_file(text, speed, out_file, voice=INWORLD_VOICE):
    """One HTTP streaming request. The answer is NDJSON (one JSON object per line); every line holds a
    base64 piece of the MP3 in result.audioContent. Each piece is decoded on its own and appended to
    out_file, so the pieces together make one playable MP3. Returns the number of audio bytes written."""
    payload = {
        "text": text,
        "voiceId": voice,
        "modelId": INWORLD_MODEL_ID,
        "audioConfig": {"audioEncoding": "MP3", "speakingRate": min(1.5, max(0.5, float(speed or 1.0)))},
        "deliveryMode": INWORLD_DELIVERY,
    }
    if INWORLD_LANGUAGE.strip().upper() != "AUTO":  # for AUTO the language field is left out on purpose
        payload["language"] = INWORLD_LANGUAGE
    request = urllib.request.Request(
        INWORLD_URL,
        data=json.dumps(payload).encode("utf-8"),
        method="POST",
        headers={
            "Authorization": "Basic " + inworld_key(),  # the key is sent exactly as it is, not encoded again
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": "PindiStudio/1.0",
        },
    )
    try:
        response = urllib.request.urlopen(request, timeout=INWORLD_TIMEOUT)
    except urllib.error.HTTPError as e:
        try:
            body = e.read()
        except Exception:
            body = b""
        raise InworldError(_inworld_http_error(e.code, body))
    except urllib.error.URLError:
        raise InworldError("Could not reach Inworld. Check your internet connection and try again.")
    except OSError:
        raise InworldError("The connection to Inworld timed out or was blocked. Check your internet and try again.")

    written = 0
    try:
        with response:
            for raw in response:  # one NDJSON line at a time, as it arrives
                line = raw.strip()
                if not line:
                    continue
                try:
                    message = json.loads(line)
                except ValueError:
                    raise InworldError("Inworld sent a damaged streaming message. Please try again.")
                if not isinstance(message, dict):
                    continue
                if message.get("error"):
                    err = message["error"]
                    detail = str(err.get("message", "")) if isinstance(err, dict) else str(err)
                    raise InworldError("Inworld stopped the speech: " + (detail or "unknown streaming error"))
                result = message.get("result")
                audio_b64 = result.get("audioContent") if isinstance(result, dict) else None
                if not audio_b64:  # e.g. a timestamp-only or usage-only message
                    continue
                try:
                    audio = base64.b64decode(audio_b64, validate=True)
                except Exception:
                    raise InworldError("Inworld sent audio that could not be decoded. Please try again.")
                out_file.write(audio)
                written += len(audio)
    except InworldError:
        raise
    except (OSError, http.client.HTTPException):
        raise InworldError("The connection to Inworld was interrupted while the audio was streaming. Please try again.")
    return written


def _looks_like_mp3(path):
    try:
        with open(path, "rb") as f:
            head = f.read(8192)
    except OSError:
        return False
    if head.startswith(b"ID3"):
        return True
    return any(head[i] == 0xFF and (head[i + 1] & 0xE0) == 0xE0 for i in range(len(head) - 1))


def inworld_script(text):
    """Split the text at <break time="..." /> tags: [("text", str) | ("pause", seconds)].
    Long pieces of text are cut into parts Inworld accepts."""
    items, pos = [], 0

    def add_text(segment):
        segment = segment.strip()
        if not segment:
            return
        if _utf16_len(segment) <= INWORLD_MAX_UNITS:
            items.append(("text", segment))
            return
        parts = split_text(segment, 3500)
        if any(_utf16_len(x) > INWORLD_MAX_UNITS for x in parts):
            parts = split_text(segment, 1500)
        items.extend(("text", x) for x in parts)

    for m in BREAK_RE.finditer(text):
        add_text(text[pos:m.start()])
        items.append(("pause", break_seconds(m.group(1))))
        pos = m.end()
    add_text(text[pos:])
    return items


def inworld_generate(text, speed, save_dir, voice=None, audio_name=None):
    """Text to speech with Inworld 360. Generator with the same outputs as clone_voice.
    <break time="1500ms" /> tags become real silence of that length (1.5 seconds)."""
    if not text or not text.strip():
        raise gr.Error("Please paste the text you want the voice to say.")
    if len(text) > MAX_TTS_CHARS:
        raise gr.Error(f"Please keep the text under {MAX_TTS_CHARS:,} characters (you pasted {len(text):,}).")
    if not (save_dir or "").strip().strip('"'):  # check before generating, so no credits are used for nothing
        raise gr.Error("Please type a save folder path or choose one with Browse.")
    inworld_key()  # clear message right away if the key is missing
    voice = inworld_voice_id(voice)
    script = inworld_script(text)
    texts = [v for k, v in script if k == "text"]
    if not texts:
        raise gr.Error("Please paste the text you want the voice to say.")
    has_pause = any(k == "pause" for k, _ in script)

    started = time.time()
    out = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False).name
    work = Path(tempfile.mkdtemp(prefix="zynth_iw_"))
    try:
        if not has_pause:  # plain text: the MP3 from Inworld is used as it is
            with open(out, "wb") as f:
                for n, part in enumerate(texts, 1):
                    yield None, "", f"Generating speech with Inworld 360 - part {n} of {len(texts)}...  ({time.time() - started:.0f}s so far)"
                    if inworld_stream_to_file(part, speed, f, voice) == 0:
                        raise InworldError("Inworld did not send any audio. Please try again.")
            if not _looks_like_mp3(out):
                raise InworldError("Inworld returned audio that is not a valid MP3. Please try again.")
        else:  # every spoken piece is made on its own, then real silence is put in between
            seq, n = [], 0
            for kind, value in script:
                if kind == "pause":
                    seq.append(float(value))
                    continue
                n += 1
                yield None, "", f"Generating speech with Inworld 360 - part {n} of {len(texts)}...  ({time.time() - started:.0f}s so far)"
                mp3 = str(work / f"part_{n:03d}.mp3")
                with open(mp3, "wb") as f:
                    if inworld_stream_to_file(value, speed, f, voice) == 0:
                        raise InworldError("Inworld did not send any audio. Please try again.")
                if not _looks_like_mp3(mp3):
                    raise InworldError("Inworld returned audio that is not a valid MP3. Please try again.")
                wav = str(work / f"part_{n:03d}.wav")
                ok, err = run_ffmpeg(["-i", mp3, "-ac", "1", "-ar", "44100", "-c:a", "pcm_s16le", wav])
                if not ok:
                    raise InworldError("Could not prepare the audio parts (ffmpeg): " + err[-200:])
                seq.append(wav)
            yield None, "", "Putting the parts and the pauses together..."
            joined = str(work / "joined.wav")
            join_wavs(seq, joined, gap_seconds=0.0)
            ok, err = run_ffmpeg(["-i", joined, "-c:a", "libmp3lame", "-q:a", "2", out])
            if not ok or not _looks_like_mp3(out):
                raise InworldError("Could not write the final MP3 (ffmpeg): " + err[-200:])
    except InworldError as e:
        try:
            os.remove(out)
        except OSError:
            pass
        raise gr.Error(str(e))
    except gr.Error:
        raise
    except Exception as e:
        try:
            os.remove(out)
        except OSError:
            pass
        raise gr.Error(f"Inworld 360 could not make the speech ({type(e).__name__}). Please try again.")
    finally:
        shutil.rmtree(work, ignore_errors=True)
    saved = save_result(out, "voiceover_inworld_" + clean_name(voice), save_dir, ext=".mp3", exact_name=audio_name)
    store_setting("inworld_voice", voice)  # remembered for next time
    took = time.time() - started
    gr.Info(f"Done in {took:.0f} seconds. Result saved.")
    yield out, saved, f"Done - Inworld 360 ({voice}), {len(texts)} part(s) in {took:.0f} seconds."


def clone_voice(sample_file, sample_mic, saved_choice, text, language, speed, model_choice, steps, clean, vc_engine, save_dir, inworld_voice=None, audio_name=None):
    """Text to speech in the chosen voice. Generator: it reports each part as it is made."""
    if not clean_name(audio_name):  # nothing is generated (no credits used) until the audio has a name
        raise gr.Error("Please write the audio name first (the 'Audio name' box above the save folder), e.g. Audio 1.")
    if model_choice == INWORLD_MODEL:  # online voice: needs no voice sample and no local engine
        yield from inworld_generate(text, speed, save_dir, inworld_voice, audio_name)
        return
    sample = get_voice_input(sample_file, sample_mic, saved_choice, "a voice sample (5-30 seconds)")
    if not text or not text.strip():
        raise gr.Error("Please paste the text you want the voice to say.")
    if len(text) > MAX_TTS_CHARS:
        raise gr.Error(f"Please keep the text under {MAX_TTS_CHARS:,} characters (you pasted {len(text):,}).")
    match_mode = model_choice == MATCH_MODEL
    pocket = model_choice == POCKET_MODEL
    fast = model_choice == FAST_MODEL
    max_match = model_choice == MAX_MODEL
    if match_mode and vc_engine == SEEDVC_LABEL and not seedvc_ready():
        raise gr.Error("Seed-VC is not installed yet. Pick another match engine (OpenVoice v2 works well).")
    if max_match and not seedvc_ready():
        raise gr.Error(
            "Max match needs Seed-VC, which is not installed yet. Close the app, run setup_seedvc.bat "
            "once, then start the app again. (Or pick the XTTS-v2 model for now.)"
        )
    if pocket and LANGUAGES[language] != "en":
        raise gr.Error("The Pocket TTS model speaks English only. Choose English, or switch to the XTTS-v2 model.")
    if fast and LANGUAGES[language] not in FAST_LANGS:
        raise gr.Error("The real-time model supports English, French and Portuguese only. "
                       "Choose one of those, or switch to the XTTS-v2 model.")
    script = parse_script(text)
    chunks = [value for kind, value in script if kind == "text"]
    if not chunks:
        raise gr.Error("Please paste the text you want the voice to say.")

    started = time.time()
    where = "GPU" if DEVICE == "cuda" else "CPU"
    if clean:
        yield None, "", "Cleaning background noise from the voice sample..."
        sample = clean_sample(sample)
    yield None, "", f"Loading the voice engine on {where} (the first time can take a minute)..."
    tts = None if match_mode else get_pocket() if pocket else get_fast_tts() if fast else get_tts()
    work = Path(tempfile.mkdtemp(prefix="zynth_tts_"))
    seq = []
    state = {}
    n = 0
    try:
        for kind, value in script:
            if kind == "pause":
                seq.append(float(value))
                continue
            n += 1
            chunk = value
            yield None, "", f"Generating part {n} of {len(chunks)} on {where}...  ({time.time() - started:.0f}s so far)"
            part = str(work / f"part_{n:03d}.wav")
            if match_mode:
                make_base_part(language, chunk, part)
            elif pocket:
                import soundfile as sf

                audio = tts.generate_audio(pocket_voice(tts, sample), chunk)
                sf.write(part, audio.detach().cpu().numpy(), int(tts.sample_rate))
            elif fast:
                tts.tts_to_file(text=chunk, speaker_wav=sample, language=FAST_LANGS[LANGUAGES[language]], file_path=part)
            else:
                make_part_xtts(tts, state, sample, chunk, language, speed, part)
            seq.append(part)
        out = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
        stretch = match_mode or pocket or fast or state.get("stretch")
        if stretch:  # the whole file is sped up afterwards, so pre-scale pauses to keep them exact
            seq = [x if isinstance(x, str) else x * float(speed) for x in seq]
        join_wavs(seq, out)
        if stretch:
            change_tempo(out, float(speed))
    finally:
        shutil.rmtree(work, ignore_errors=True)
    if match_mode:
        yield None, "", (
            f"Base voice ready in {time.time() - started:.0f}s. Now turning it into your sample's voice "
            "(the first time it also downloads/loads the match engine)..."
        )
        out = convert_voice(vc_engine, out, sample, steps)
    if max_match:
        yield None, "", (
            f"Voice generated in {time.time() - started:.0f}s. Now matching it to your sample with Seed-VC "
            "(the slowest step; the first time it also loads its models)..."
        )
        out = run_seedvc(out, sample, steps)
        if clean:
            out = run_filter(out, POLISH_FILTER, rate="44100")
    saved = save_result(out, "voiceover", save_dir, exact_name=audio_name)
    took = time.time() - started
    gr.Info(f"Done in {took:.0f} seconds. Result saved.")
    yield out, saved, f"Done - {len(chunks)} part(s) in {took:.0f} seconds on {where}."


def wav_seconds(path):
    import wave

    try:
        with wave.open(path, "rb") as w:
            return w.getnframes() / float(w.getframerate())
    except Exception:
        return 0.0


def shorten_target(path, limit=30.0, keep=30):
    """A very long target sample (minutes of audio, often with pauses, music or other voices)
    blurs the voice 'fingerprint'. Keep the first `keep` seconds of actual speech instead."""
    if wav_seconds(path) <= limit:
        return path
    exe = get_ffmpeg()
    if not exe:
        return path
    out = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
    filt = "silenceremove=start_periods=1:start_threshold=-45dB:stop_periods=-1:stop_duration=0.5:stop_threshold=-45dB"
    cmd = [exe, "-y", "-i", path, "-af", filt, "-t", str(keep), "-ar", "24000", "-ac", "1", out]
    try:
        r = subprocess.run(cmd, capture_output=True, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        if r.returncode == 0 and wav_seconds(out) >= 5:
            return out
    except Exception:
        pass
    return path


def split_audio(path, seconds=20):
    """Cut a long WAV into pieces of about `seconds` seconds. Returns the list of piece files."""
    if wav_seconds(path) <= seconds * 1.5:
        return [path]
    exe = get_ffmpeg()
    if not exe:
        return [path]
    folder = tempfile.mkdtemp(prefix="zynth_split_")
    cmd = [exe, "-y", "-i", path, "-f", "segment", "-segment_time", str(seconds),
           "-c:a", "pcm_s16le", os.path.join(folder, "part_%03d.wav")]
    try:
        r = subprocess.run(cmd, capture_output=True, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    except Exception:
        return [path]
    parts = sorted(str(f) for f in Path(folder).glob("part_*.wav"))
    return parts if r.returncode == 0 and parts else [path]


def concat_audio(parts, out_path):
    """Join WAV files of the same format end to end (also reads float WAVs)."""
    listing = out_path + ".txt"
    with open(listing, "w", encoding="utf-8") as f:
        for part in parts:
            f.write("file '" + part.replace("\\", "/").replace("'", "'\\''") + "'\n")
    cmd = [get_ffmpeg(), "-y", "-f", "concat", "-safe", "0", "-i", listing, "-c:a", "pcm_s16le", out_path]
    r = subprocess.run(cmd, capture_output=True, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    try:
        os.remove(listing)
    except OSError:
        pass
    if r.returncode != 0 or not os.path.exists(out_path):
        raise gr.Error("Could not join the converted parts together.")


MAX_VC_FILES = 6
MAX_TARGET_SECONDS = 30  # the target voice sample is never longer than this; the original audio can be any length


def progress_bar(pct, width=20):
    done = int(round(width * max(0.0, min(100.0, pct)) / 100.0))
    return "[" + "#" * done + "-" * (width - done) + "]"


def estimate_percent(elapsed, expected):
    """The models give no real progress, so the percentage is an estimate from how long this PC
    took for earlier conversions. It never shows 100% before the work is really finished."""
    import math

    if elapsed <= expected:
        return 90.0 * elapsed / expected
    return 90.0 + 9.0 * (1.0 - math.exp(-(elapsed - expected) / expected))


class VCStopped(Exception):
    pass


def convert_one(source, target, model_choice, steps, note, cancel=None):
    """Convert one prepared WAV of any length in ONE piece (no cutting, so no seams in the result).
    Generator: yields ("status", (percent, text)) every second or two, finally ("done", out_path)."""
    started = time.time()
    seed = model_choice == SEEDVC_LABEL
    result = {}
    seconds = max(1.0, wav_seconds(source))
    rate_key = f"vc_rate_seedvc_{int(steps)}" if seed else f"vc_rate_{model_choice}"
    rate = float(load_setting(rate_key, 0.4 * int(steps) if seed else 0.6))  # processing seconds per audio second
    expected = max(10.0, seconds * rate + 10.0)

    def work():
        try:
            if seed:
                result["out"] = run_seedvc(source, target, steps)
            else:
                out = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
                get_vc(model_choice).voice_conversion_to_file(source_wav=source, target_wav=target, file_path=out)
                result["out"] = out
        except BaseException as e:
            result["err"] = e

    th = threading.Thread(target=work, daemon=True)
    th.start()
    hint = "the first time it also loads the model" + ("; Seed-VC can take several minutes" if seed else "")
    while th.is_alive():
        th.join(1.0)
        if cancel is not None and cancel.is_set():
            raise VCStopped()  # the worker thread is left to finish on its own; its result is ignored
        if th.is_alive():
            elapsed = time.time() - started
            yield "status", (estimate_percent(elapsed, expected), f"{elapsed:.0f}s so far ({hint})")
    if "err" in result:
        raise result["err"]
    took = time.time() - started
    try:  # remember how fast this PC is, so the next percentage is more accurate
        measured = took / seconds
        store_setting(rate_key, round(0.5 * rate + 0.5 * measured, 4) if load_setting(rate_key, None) else round(measured, 4))
    except Exception:
        pass
    yield "done", result["out"]


def save_named(tmp_path, folder, source_path, ext=".wav", tag="changed voice"):
    """Save a converted file in the chosen folder under the SAME name as the original audio
    (original "audio 1.mp3" -> "audio 1.wav"). If that name is taken, or it would replace the
    original itself, " changed voice" is added so nothing is ever overwritten."""
    folder = (folder or "").strip().strip('"')
    if not folder:
        raise gr.Error("Please type a save folder path or choose one with Browse.")
    stem = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", Path(source_path).stem).strip(" .") or "audio"
    original = os.path.normcase(os.path.abspath(str(source_path)))
    try:
        Path(folder).mkdir(parents=True, exist_ok=True)
        final = Path(folder) / f"{stem}{ext}"
        n = 1
        while final.exists() or os.path.normcase(os.path.abspath(str(final))) == original:
            final = Path(folder) / (f"{stem} {tag}{ext}" if n == 1 else f"{stem} {tag} ({n}){ext}")
            n += 1
        shutil.copy(tmp_path, final)
    except OSError as e:
        raise gr.Error(f"Could not save to this folder: {e}")
    store_setting("save_dir", folder)
    return str(final)


VC_ROWS = MAX_VC_FILES


def vc_progress_cell(pct):
    """Progress column of one row: nothing (empty row), 'Queue', or a percentage with a small bar."""
    base = "text-align:center;padding:10px 4px;font-size:15px;"
    if pct is None:
        return f'<div style="{base}"></div>'
    if pct == "queue":
        return f'<div style="{base}opacity:.65">Queue</div>'
    bar = ('<div style="height:5px;border-radius:4px;background:rgba(128,128,128,.3);margin-top:5px;overflow:hidden">'
           f'<div style="height:100%;width:{pct:.0f}%;background:linear-gradient(90deg,#0f766e,#ff7a59)"></div></div>')
    return f'<div style="{base}"><b>{pct:.0f}%</b>{bar}</div>'


def vc_status_cell(text):
    from html import escape

    color = {"Done": "#22c55e", "Failed": "#ef4444", "Stopped": "#f59e0b"}.get(text.split(":")[0], "inherit")
    return (f'<div style="text-align:center;padding:10px 4px;font-size:15px;font-weight:600;color:{color}">'
            f"{escape(text)}</div>")


def vc_overall(pct):
    return f'<div style="font-weight:600;margin:4px 0">Overall: {pct:.0f}%</div>'


_vc_stop = [threading.Event() for _ in range(VC_ROWS)]
_vc_state = {"row": None, "seed": False}


def vc_stop_row(row):
    """Stop button of one row: a waiting row is skipped; the running row is abandoned (Seed-VC is restarted)."""
    _vc_stop[row].set()
    if _vc_state["row"] == row and _vc_state["seed"]:
        _sv_stop()  # ends the Seed-VC worker; it starts again by itself for the next conversion


def make_stop_handler(row):
    def handler():
        vc_stop_row(row)
    return handler


def share_copy(path):
    """Gradio may only show files from the temp folder, so show a copy there (the real file stays in the save folder)."""
    d = tempfile.mkdtemp(prefix="zynth_out_")
    dest = os.path.join(d, os.path.basename(path))
    shutil.copy(path, dest)
    return dest


def change_voice(*args):
    """Six rows: each row has its own original audio and its own saved target voice.
    The rows are converted one after another. Generator: updates the table cells live.
    Arguments: 6 files, 6 voice names, model, steps, clean, save folder."""
    n_rows = VC_ROWS
    files = list(args[:n_rows])
    voices = list(args[n_rows:2 * n_rows])
    model_choice, steps, clean, save_dir = args[2 * n_rows:2 * n_rows + 4]

    jobs = []  # (row index, source path, voice name)
    for i in range(n_rows):
        path = pick_path(files[i])
        if path:
            jobs.append((i, path, voices[i]))
    if not jobs:
        raise gr.Error("Please drop at least one audio file into the Source Audio column.")

    prog = [None] * n_rows
    stat = [""] * n_rows
    for i, _, _ in jobs:
        prog[i] = "queue"
    state = {"overall": 0.0, "latest": None}
    results = []
    shown = []
    for ev in _vc_stop:
        ev.clear()
    log = [f"Model: {model_choice}", ""]
    total = len(jobs)
    started = time.time()

    def show():
        return (state["latest"], "\n".join(log), (shown or None), vc_overall(state["overall"]),
                *[vc_progress_cell(p) for p in prog], *[vc_status_cell(t) for t in stat])

    yield show()
    targets = {}
    for n, (i, path, voice) in enumerate(jobs, 1):
        label = os.path.basename(path)
        log.append(f"[{n}/{total}] row {i + 1}: {label}  ->  voice: {voice or '(none)'}")
        file_start = time.time()
        cancel = _vc_stop[i]
        if cancel.is_set():
            stat[i], prog[i] = "Stopped", None
            log.append("   stopped before it started")
            log.append("")
            yield show()
            continue
        _vc_state["row"], _vc_state["seed"] = i, model_choice == SEEDVC_LABEL
        try:
            if not voice or voice == NO_SAVED or not saved_voice_path(voice):
                raise gr.Error("Select a target voice in this row first.")
            if voice not in targets:
                t = shorten_target(saved_voice_path(voice), limit=MAX_TARGET_SECONDS, keep=MAX_TARGET_SECONDS)
                targets[voice] = clean_sample(t) if clean else t
            target = targets[voice]
            prog[i], stat[i] = 0.0, "Working..."
            state["overall"] = (n - 1) * 100 / total
            source = to_wav(path)
            if clean:
                yield show()
                source = run_filter(source, SOURCE_CLEAN_FILTER, rate=None)
            if cancel.is_set():
                raise VCStopped()
            out = None
            for kind, value in convert_one(source, target, model_choice, steps, "", cancel):
                if kind == "status":
                    pct, text = value
                    prog[i] = pct
                    state["overall"] = ((n - 1) * 100 + pct) / total
                    yield show()
                else:
                    out = value
            if clean:
                out = run_filter(out, POLISH_FILTER, rate=None)
            saved = save_named(out, save_dir, path)
            results.append(saved)
            state["latest"] = share_copy(saved)
            shown.append(state["latest"])
            prog[i], stat[i] = 100.0, "Done"
            state["overall"] = n * 100 / total
            log.append(f"   100% in {time.time() - file_start:.0f}s, saved: {saved}")
        except VCStopped:
            stat[i], prog[i] = "Stopped", None
            log.append("   stopped by you")
        except Exception as e:  # one bad row must not stop the others
            if cancel.is_set():  # the stop button killed the engine, so the error is expected
                stat[i], prog[i] = "Stopped", None
                log.append("   stopped by you")
            else:
                reason = getattr(e, "message", None) or str(e)
                stat[i] = "Failed"
                prog[i] = prog[i] if isinstance(prog[i], float) else None
                log.append(f"   X failed: {reason}")
        _vc_state["row"] = None
        log.append("")
        yield show()

    state["overall"] = 100.0
    summary = f"{len(results)}/{total} converted in {time.time() - started:.0f} seconds."
    log.append("-" * 40)
    log.append(summary)
    gr.Info(summary)
    yield show()


# ---------- ASMR: remove music, keep the original sound ----------
ASMR_WORKER_CODE = r'''
import os, sys, traceback


def suppress_lyrics(x, sr, level):
    # x: (samples, channels). Removes the quiet singing that sits behind a louder voiceover:
    # it learns what the quiet moments (pauses of the voiceover) sound like and takes that away.
    import numpy as np
    from scipy.signal import stft, istft
    from scipy.ndimage import maximum_filter1d, uniform_filter1d
    nper, hop = 2048, 512
    mono = x.mean(axis=1)
    _, _, Zm = stft(mono, fs=sr, nperseg=nper, noverlap=nper - hop, window="hann")
    power = np.abs(Zm) ** 2
    e = 10 * np.log10(power.mean(axis=0) + 1e-12)
    ref = np.percentile(e, 95)
    gap = 18.0 if level == 1 else 13.0
    quiet = e < ref - gap
    over = 2.0 if level == 1 else 3.5
    if quiet.sum() >= 5:
        noise = power[:, quiet].mean(axis=1, keepdims=True)
    else:
        noise = np.zeros((power.shape[0], 1))
    gain = np.sqrt(np.clip(1.0 - over * noise / (power + 1e-12), 0.05, 1.0))
    frame_gain = np.where(e < ref - gap - 3, 0.03, 1.0)
    hold = max(1, int(0.25 * sr / hop))
    frame_gain = uniform_filter1d(maximum_filter1d(frame_gain, size=hold), size=5)
    gain = uniform_filter1d(gain * frame_gain[None, :], size=3, axis=1)
    out = np.zeros_like(x)
    for c in range(x.shape[1]):
        _, _, Zc = stft(x[:, c], fs=sr, nperseg=nper, noverlap=nper - hop, window="hann")
        _, y = istft(Zc * gain, fs=sr, nperseg=nper, noverlap=nper - hop, window="hann")
        n = min(len(y), x.shape[0])
        out[:n, c] = y[:n]
    return out.astype(np.float32)


def percussive(rest_np, sr, high):
    # short, noisy sounds (taps, crinkles, brushing) - not the sustained, tonal song
    import numpy as np
    import librosa
    from scipy.signal import butter, sosfilt
    sos = butter(6 if high else 4, 2500 if high else 150, btype="highpass", fs=sr, output="sos")
    perc = []
    for ch in rest_np:
        _, p = librosa.effects.hpss(ch, margin=8.0 if high else 3.0)
        perc.append(sosfilt(sos, p).astype("float32"))
    return np.stack(perc)


def main():
    src, out_dir, keep, mode, passes, model_name, debug, lyrics = (
        sys.argv[1], sys.argv[2], float(sys.argv[3]), sys.argv[4], int(sys.argv[5]), sys.argv[6],
        sys.argv[7] == "1", int(sys.argv[8]))
    try:
        import numpy as np
        import soundfile as sf
        import torch
        from demucs.pretrained import get_model
        from demucs.apply import apply_model
    except Exception as e:
        print("ERR missing component: %r" % (e,))
        return
    try:
        torch.set_num_threads(max(1, os.cpu_count() or 2))
        model = get_model(model_name)
        model.eval()
        sr = model.samplerate
        data, file_sr = sf.read(src, dtype="float32", always_2d=True)
        if file_sr != sr:
            print("ERR unexpected sample rate %s" % file_sr)
            return
        orig = torch.from_numpy(np.ascontiguousarray(data.T))
        if orig.shape[0] == 1:
            orig = orig.repeat(2, 1)
        idx = model.sources.index("vocals")
        shifts = 1 if model_name not in ("htdemucs", "htdemucs_6s") else 0

        def write(name, arr):
            arr = np.asarray(arr, dtype="float32")
            peak = float(np.abs(arr).max())
            if peak > 0.99:
                arr = arr * (0.99 / peak)
            sf.write(os.path.join(out_dir, name), arr, sr, subtype="PCM_16")

        def run_model(sig):
            ref = sig.mean(0)
            mean, std = ref.mean(), ref.std() + 1e-8
            with torch.no_grad():
                out = apply_model(model, ((sig - mean) / std)[None], device="cpu", shifts=shifts,
                                  split=True, overlap=0.25, progress=False)[0]
            return out * std + mean

        if mode == "split":
            out = run_model(orig)
            stems = {name: out[i] for i, name in enumerate(model.sources)}
            vocals = stems.pop("vocals")
            v_np = vocals.T.numpy()
            voice_np = suppress_lyrics(v_np, sr, max(1, lyrics))
            write("stem_voiceover.wav", voice_np)
            write("stem_singing.wav", v_np - voice_np)
            rest = sum(stems.values())
            for name, arr in stems.items():
                write("stem_" + name + ".wav", arr.T.numpy())
            try:
                write("stem_asmr_sounds.wav", percussive(rest.numpy(), sr, True).T)
            except Exception as e:
                print("note: asmr sounds part skipped: %r" % (e,), file=sys.stderr)
            print("OK " + out_dir)
            return

        first = {}

        def separate(sig):
            out = run_model(sig)
            vocals = out[idx]
            if not first:
                first["voice"] = vocals.T.numpy().copy()
                first["other"] = (out.sum(0) - vocals).T.numpy().copy()
            if lyrics:
                vocals = torch.from_numpy(np.ascontiguousarray(suppress_lyrics(vocals.T.numpy(), sr, lyrics).T))
            kept = vocals
            if mode in ("asmr", "asmr_high"):
                kept = vocals + torch.from_numpy(percussive((out.sum(0) - out[idx]).numpy(), sr, mode == "asmr_high"))
            return kept

        cur = orig
        for _ in range(passes):  # a second pass cleans the little music that leaked into the first result
            cur = separate(cur)
        mix = (cur + keep * (orig - cur)).T.numpy()
        write("clean.wav", mix)
        if debug:
            for name, arr in first.items():
                write("part_" + name + ".wav", arr)
        print("OK " + os.path.join(out_dir, "clean.wav"))
    except Exception as e:
        traceback.print_exc()
        print("ERR %r" % (e,))


main()
'''
_ASMR_LOG = Path(tempfile.gettempdir()) / "zynth_asmr_worker.log"


def _clean_python_env():
    env = dict(os.environ, PYTHONUTF8="1", PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1")
    for var in ("PYTHONHOME", "PYTHONPATH"):
        env.pop(var, None)
    return env


def asmr_check_ready():
    if not SEEDVC_PY.exists():
        raise gr.Error(
            "The ASMR tab uses the Seed-VC Python environment, which was not found. "
            "Run setup_seedvc.bat once (or use the Seed-VC build of the app), then run install_demucs.bat."
        )
    flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    try:
        r = subprocess.run([str(SEEDVC_PY), "-c", "import demucs.pretrained, demucs.apply, soundfile, torch"], capture_output=True, text=True,
                           timeout=120, env=_clean_python_env(), creationflags=flags)
    except (OSError, subprocess.TimeoutExpired) as e:
        raise gr.Error(f"Could not start the ASMR engine: {e}")
    if r.returncode != 0:
        last = (r.stderr or "").strip().splitlines()[-1:] or [""]
        raise gr.Error("The music remover (Demucs) is not installed yet. Close the app, run install_demucs.bat once, "
                       f"then start the app again.\n{last[0]}")


def media_info(path):
    """(has_video, seconds) read from ffmpeg's own report."""
    exe = get_ffmpeg()
    flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    r = subprocess.run([exe, "-hide_banner", "-i", path], capture_output=True, text=True, errors="replace", creationflags=flags)
    text = r.stderr or ""
    has_video = any("Video:" in line and "attached pic" not in line for line in text.splitlines())
    m = re.search(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)", text)
    seconds = int(m.group(1)) * 3600 + int(m.group(2)) * 60 + float(m.group(3)) if m else 0.0
    return has_video, seconds


def run_ffmpeg(args):
    exe = get_ffmpeg()
    flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    r = subprocess.run([exe, "-y", "-hide_banner", "-loglevel", "error", *args], capture_output=True, text=True,
                       errors="replace", creationflags=flags)
    return r.returncode == 0, (r.stderr or "").strip()


def asmr_remove_music(files, mode_choice, strength_choice, lyrics_choice, keep_pct, save_dir, debug_parts=False):
    paths = [pick_path(f) for f in (files or [])]
    paths = [p for p in paths if p]
    if not paths:
        raise gr.Error("Please add at least one audio or video file.")
    if len(paths) > 6:
        raise gr.Error("Please add at most 6 files at a time.")
    if not (save_dir or "").strip():
        raise gr.Error("Please type a save folder path or choose one with Browse.")
    yield "Checking the music remover...", None
    asmr_check_ready()
    keep = max(0.0, min(1.0, float(keep_pct or 0) / 100.0))
    mode_text = str(mode_choice or "")
    if mode_text.startswith("Separate everything"):
        mode = "split"
    elif mode_text.startswith("Voiceover only"):
        mode = "voice"
    else:
        mode = "asmr" if "all ASMR" in mode_text else "asmr_high"
    lyrics_text = str(lyrics_choice or "")
    lyrics = 0 if lyrics_text.startswith("Off") else (2 if lyrics_text.startswith("Strong") else 1)
    strength = str(strength_choice or "")
    passes = 1 if strength.startswith("Normal") else 2
    model_name = "htdemucs_ft" if strength.startswith("Maximum") else "htdemucs"
    if mode == "split":  # six parts (adds guitar and piano) unless the best 4-part model was chosen
        passes = 1
        if model_name == "htdemucs":
            model_name = "htdemucs_6s"
    worker = Path.home() / ".zynth_studio" / "asmr_worker.py"
    worker.parent.mkdir(parents=True, exist_ok=True)
    worker.write_text(ASMR_WORKER_CODE, encoding="utf-8")
    flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    total = len(paths)
    lines, latest_audio = [], None
    started = time.time()

    def text(live=None):
        return "\n".join(lines + ([live] if live else []))

    for n, path in enumerate(paths, 1):
        label = os.path.basename(path)
        lines.append(f"[{n}/{total}] {label}")
        work = Path(tempfile.mkdtemp(prefix="zynth_asmr_"))
        try:
            has_video, seconds = media_info(path)
            yield text("   reading the audio..."), latest_audio
            ok, err = run_ffmpeg(["-i", path, "-vn", "-ac", "2", "-ar", "44100", "-c:a", "pcm_s16le", str(work / "in.wav")])
            if not ok:
                raise RuntimeError("Could not read the audio of this file. " + err[-200:])
            result = {}

            def job():
                try:
                    with open(_ASMR_LOG, "w", encoding="utf-8") as log:
                        r = subprocess.run([str(SEEDVC_PY), str(worker), str(work / "in.wav"), str(work), str(keep), mode, str(passes), model_name, "1" if debug_parts else "0", str(lyrics)],
                                           stdout=subprocess.PIPE, stderr=log, text=True, encoding="utf-8",
                                           env=_clean_python_env(), creationflags=flags)
                    result["out"] = (r.stdout or "").strip().splitlines()[-1:] or [""]
                except Exception as e:
                    result["out"] = [f"ERR {e}"]

            th = threading.Thread(target=job, daemon=True)
            th.start()
            t0 = time.time()
            expected = 25 + max(seconds, 10) * (2.0 if mode == "asmr" else 1.5) * passes * (4 if model_name != "htdemucs" else 1)
            while th.is_alive():
                th.join(1.0)
                pct = estimate_percent(time.time() - t0, expected)
                yield text(f"   removing music... {pct:.0f}% (the first time also downloads the model)"), latest_audio
            reply = result.get("out", [""])[0]
            if mode == "split" and reply.startswith("OK "):
                names = {"voiceover": "voiceover", "singing": "singing (lyrics)", "drums": "drums", "bass": "bass",
                         "other": "other instruments", "guitar": "guitar", "piano": "piano",
                         "asmr_sounds": "ASMR sounds (taps, crinkles)"}
                folder_out = Path(save_dir.strip().strip('"')) / f"{Path(path).stem} - parts"
                folder_out.mkdir(parents=True, exist_ok=True)
                for part in sorted(work.glob("stem_*.wav")):
                    key = part.stem[len("stem_"):]
                    dest = folder_out / f"{Path(path).stem} - {names.get(key, key)}.wav"
                    shutil.copy(part, dest)
                    if key == "voiceover":
                        latest_audio = share_copy(str(dest))
                    lines.append(f"   saved: {dest}")
                store_setting("save_dir", save_dir.strip().strip('"'))
                lines.append("   100% - all parts saved in: " + str(folder_out))
                lines.append("")
                yield text(), latest_audio
                continue
            if not reply.startswith("OK "):
                tail = ""
                try:
                    tail = _ASMR_LOG.read_text(encoding="utf-8", errors="replace").strip()[-300:]
                except OSError:
                    pass
                raise RuntimeError((reply[4:] or "the music remover stopped") + (("\n   " + tail) if tail else ""))
            clean_wav = reply[3:].strip()
            if has_video:
                lines.append("   joining the clean sound with the video...")
                yield text(), latest_audio
                mixed = None
                for ext, acodec in ((".mp4", "aac"), (".mkv", "aac")):
                    out_file = work / f"video{ext}"
                    ok, err = run_ffmpeg(["-i", path, "-i", clean_wav, "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy",
                                          "-c:a", acodec, "-b:a", "192k", "-shortest", str(out_file)])
                    if ok and out_file.exists():
                        mixed = (out_file, ext)
                        break
                if not mixed:
                    raise RuntimeError("Could not put the new sound into the video. " + err[-200:])
                saved = save_named(str(mixed[0]), save_dir, path, ext=mixed[1], tag="no music")
            else:
                saved = save_named(clean_wav, save_dir, path, ext=".wav", tag="no music")
                latest_audio = share_copy(saved)
            lines.append(f"   100% - saved: {saved}")
            if debug_parts:
                stem = Path(path).stem
                for part, label_part in (("part_voice.wav", "voice part"), ("part_other.wav", "other part")):
                    if (work / part).exists():
                        dest = Path(save_dir.strip().strip('"')) / f"{stem} {label_part}.wav"
                        shutil.copy(work / part, dest)
                        lines.append(f"   check file: {dest}")
        except Exception as e:
            lines.append(f"   X failed: {getattr(e, 'message', None) or e}")
        finally:
            shutil.rmtree(work, ignore_errors=True)
        lines.append("")
        yield text(), latest_audio
    lines.append("-" * 40)
    lines.append(f"Finished in {time.time() - started:.0f} seconds.")
    gr.Info("ASMR music removal finished.")
    yield text(), latest_audio


# ---------- Downloader (YouTube, TikTok, Instagram and 1000+ other sites, via yt-dlp) ----------
ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
DL_FORMATS = {
    "Audio - MP3": ("mp3", "audio"),
    "Audio - WAV": ("wav", "audio"),
    "Video - MP4": ("mp4", "video"),
}
MAX_LINKS = 50


def ffmpeg_dir():
    """yt-dlp wants a folder holding a file literally named ffmpeg(.exe); make one from the bundled ffmpeg."""
    exe = get_ffmpeg()
    if not exe:
        raise gr.Error("ffmpeg was not found. Close the app and run run.bat again to install it.")
    folder = Path.home() / ".zynth_studio" / "ffmpeg"
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / ("ffmpeg.exe" if os.name == "nt" else "ffmpeg")
    try:
        if not target.exists() or target.stat().st_size != os.path.getsize(exe):
            shutil.copy2(exe, target)
    except OSError as e:
        raise gr.Error(f"Could not prepare ffmpeg: {e}")
    return str(folder)


def parse_input(text):
    """Optional folder name = any text before the first link. Returns (folder_name, unique links)."""
    lines = (text or "").splitlines()
    first = next((i for i, ln in enumerate(lines) if re.search(r"https?://", ln, re.I)), None)
    if first is None:
        return "", []
    name = " ".join(ln.strip() for ln in lines[:first] if ln.strip())
    links = []
    for part in re.split(r"\s+", "\n".join(lines[first:])):
        part = part.strip()
        if part and part not in links:
            links.append(part)
    return name, links


_LINK_RE = re.compile(r"https?://[^\s*<>\"'`]+", re.I)


def clean_links(text):
    """Tidy pasted links: removes the stars (**), the numbering (1. 2. ...) and extra words around each link, so
    that every line is one clean link. A folder name typed before the first link is kept."""
    raw = (text or "").replace("\r", "")
    lines = raw.split("\n")
    first = next((i for i, ln in enumerate(lines) if re.search(r"https?://", ln, re.I)), None)
    if first is None:
        gr.Warning("No link found. A link must start with http:// or https://")
        return text
    name = " ".join(ln.replace("*", "").strip() for ln in lines[:first] if ln.replace("*", "").strip())
    links, left_over = [], 0
    for ln in lines[first:]:
        found = []
        for m in _LINK_RE.findall(ln):
            m = m.rstrip(",;.]")
            while m.endswith(")") and m.count(")") > m.count("("):
                m = m[:-1]
            if m:
                found.append(m)
        links.extend(found)
        rest = _LINK_RE.sub(" ", ln).replace("*", " ")
        rest = re.sub(r"(?:^|\s)\(?\d{1,3}\s*[.)\]:\-]+(?=\s|$)", " ", rest)  # the numbers (1. 2) 3: ...)
        left_over += len(re.findall(r"[^\W_]+", rest))
    cleaned = (name + "\n\n" if name else "") + "\n".join(links)
    note = f"Cleaned: {len(links)} link(s) ready, one per line, stars and numbering removed."
    if left_over:
        note += f" {left_over} extra word(s) next to the links were removed too."
    gr.Info(note)
    return cleaned


def next_number(folder, prefix):
    """First free number for names like 'audio 1', 'audio 2' (so nothing is ever overwritten)."""
    highest = 0
    pattern = re.compile(rf"^{re.escape(prefix)} (\d+)\.", re.I)
    try:
        for p in Path(folder).iterdir():
            m = pattern.match(p.name)
            if m:
                highest = max(highest, int(m.group(1)))
    except OSError:
        pass
    return highest + 1


OUTDATED_WORDS = ("unable to extract", "webpage video data", "failed to parse", "unable to find", "extractorerror")


def looks_outdated(exc):
    low = ANSI_RE.sub("", str(exc)).lower()
    return any(w in low for w in OUTDATED_WORDS)


_ytdlp_update_tried = False


def update_ytdlp():
    """Upgrade yt-dlp in this app's Python and reload it (works when started from run.bat or the
    shortcut; a built .exe has its engine packed inside, so it needs a rebuild instead)."""
    global _ytdlp_update_tried
    if getattr(sys, "frozen", False) or _ytdlp_update_tried:
        return None
    _ytdlp_update_tried = True
    try:
        r = subprocess.run(
            [sys.executable, "-m", "pip", "install", "-U", "yt-dlp[default]"],
            capture_output=True, text=True, timeout=300,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        if r.returncode != 0:
            return None
        for name in [m for m in sys.modules if m == "yt_dlp" or m.startswith("yt_dlp.")]:
            del sys.modules[name]
        import yt_dlp

        return yt_dlp
    except Exception:
        return None


def explain_error(exc):
    msg = ANSI_RE.sub("", str(exc)).strip()
    msg = re.sub(r"^ERROR:\s*", "", msg)
    msg = " ".join(msg.split())[:350]
    low = msg.lower()
    if is_bot_check(exc):
        reason = ("YouTube is asking this connection to prove it is not a bot. Try again later or on another network. "
                  "If it keeps happening, add YOUR YouTube cookies in the Downloader tab (Cookies section, site: YouTube).")
    elif any(w in low for w in ("login", "log in", "sign in", "cookie", "private", "rate-limit", "rate limit")):
        reason = ("This post needs a login, is private, or the site is limiting requests. For Instagram/TikTok posts that need "
                  "a login, add YOUR cookies.txt in the Downloader tab (Cookies section) and try again.")
    elif "ip address is blocked" in low or "blocked from accessing" in low:
        reason = "The site blocked this internet connection for this video. Try another network (or a VPN) and run it again."
    elif looks_outdated(exc):
        reason = ("The downloader could not read this page - its engine is probably out of date for this site "
                  "(TikTok changes often). Run update_downloader.bat (or rebuild the .exe after it) and try again.")
    elif any(w in low for w in ("unsupported url", "no suitable extractor")):
        reason = "This link is not supported."
    elif any(w in low for w in ("unavailable", "removed", "deleted", "not exist", "404")):
        reason = "This video is unavailable or was removed."
    elif any(w in low for w in ("urlopen error", "getaddrinfo", "timed out", "connection", "network")):
        reason = "Could not connect. Please check your internet connection."
    else:
        reason = "The download failed."
    return f"{reason}\n   Details: {msg}"


_dl_cancel = threading.Event()


class DownloadCancelled(Exception):
    pass


def cancel_download():
    """Called by the Cancel button: the running download stops at its next step."""
    _dl_cancel.set()
    gr.Info("Cancelling the download...")


def remove_partial(folder, stem):
    """Delete the half-downloaded files of a cancelled link (they are named '<stem>.*')."""
    try:
        for p in Path(folder).glob(glob_escape(stem) + ".*"):
            try:
                p.unlink()
            except OSError:
                pass
    except OSError:
        pass


def glob_escape(text):
    return re.sub(r"([\[\]*?])", r"[\1]", text)


COOKIE_SITES = {  # which cookie file name belongs to which website
    "instagram": ("instagram.com",),
    "tiktok": ("tiktok.com",),
    "youtube": ("youtube.com", "youtu.be"),
    "facebook": ("facebook.com", "fb.watch", "fb.com"),
    "twitter": ("twitter.com", "x.com"),
}
COOKIES_DIR = Path.home() / ".zynth_studio" / "cookies"


def cookie_file_for(url):
    """A cookies.txt file that YOU added for this site (Downloader tab -> Cookies), or None.
    Looks for <site>_cookies.txt (for example instagram_cookies.txt), then cookies.txt, in the cookies
    folder and next to the app."""
    host = (re.sub(r"^https?://", "", url, flags=re.I).split("/")[0].split(":")[0] or "").lower()
    names = [f"{site}_cookies.txt" for site, doms in COOKIE_SITES.items()
             if any(host == d or host.endswith("." + d) for d in doms)] + ["cookies.txt"]
    for folder in (COOKIES_DIR, BASE_DIR):
        for name in names:
            path = folder / name
            try:
                if path.is_file() and path.stat().st_size > 20:
                    return str(path)
            except OSError:
                pass
    return None


def save_cookie_file(site, file_value):
    path = pick_path(file_value)
    if not path:
        raise gr.Error("Please choose your cookies.txt file first.")
    try:
        head = Path(path).read_text(encoding="utf-8", errors="ignore")[:2000]
    except OSError as e:
        raise gr.Error(f"Could not read this file: {e}")
    if "\t" not in head and "Netscape" not in head:
        raise gr.Error("This does not look like a cookies.txt file (Netscape format). Export it again with a cookies.txt browser extension.")
    key = {"Instagram": "instagram", "TikTok": "tiktok", "YouTube": "youtube", "Facebook": "facebook", "Twitter / X": "twitter"}.get(site, "")
    name = f"{key}_cookies.txt" if key else "cookies.txt"
    try:
        COOKIES_DIR.mkdir(parents=True, exist_ok=True)
        shutil.copy(path, COOKIES_DIR / name)
    except OSError as e:
        raise gr.Error(f"Could not save the cookies: {e}")
    return f"Saved as {name}. Downloads from {site if key else 'all sites'} will now use it."


def remove_cookie_files():
    count = 0
    try:
        for f in COOKIES_DIR.glob("*cookies.txt"):
            f.unlink()
            count += 1
    except OSError as e:
        raise gr.Error(f"Could not remove them: {e}")
    return f"Removed {count} cookie file(s)."


def download_one(yt_dlp, url, codec, kind, folder, out_name, ffmpeg_loc, on_progress, extra_opts=None):
    def hook(d):
        if _dl_cancel.is_set():
            raise DownloadCancelled("Cancelled")
        status = d.get("status")
        if status == "downloading":
            total = d.get("total_bytes") or d.get("total_bytes_estimate")
            if total:
                on_progress(min(d.get("downloaded_bytes", 0) / total, 0.98))
        elif status == "finished":
            on_progress(0.99)

    opts = {
        # '%' is special in yt-dlp output templates, so escape it in the folder part
        "outtmpl": str(Path(folder)).replace("%", "%%") + os.sep + f"{out_name}.%(ext)s",
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
        "noprogress": True,
        "progress_hooks": [hook],
        "ffmpeg_location": ffmpeg_loc,
        "windowsfilenames": True,
        "retries": 3,
        "socket_timeout": 30,
    }
    if kind == "audio":
        opts["format"] = "bestaudio/best"
        opts["postprocessors"] = [
            {"key": "FFmpegExtractAudio", "preferredcodec": codec, "preferredquality": "192"}
        ]
    else:
        opts["format"] = "bv*[ext=mp4]+ba[ext=m4a]/b[ext=mp4]/bv*+ba/b"
        opts["merge_output_format"] = "mp4"

    cookie_src = cookie_file_for(url)
    cookie_copy = None
    if cookie_src:  # use a copy: yt-dlp writes cookies back to the file it is given
        try:
            cookie_copy = tempfile.NamedTemporaryFile(suffix=".txt", delete=False).name
            shutil.copy(cookie_src, cookie_copy)
            opts["cookiefile"] = cookie_copy
        except OSError:
            cookie_copy = None

    for key, value in (extra_opts or {}).items():
        if isinstance(value, dict) and isinstance(opts.get(key), dict):
            opts[key] = {**opts[key], **value}
        else:
            opts[key] = value

    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(url, download=True)
    finally:
        if cookie_copy:
            try:
                os.unlink(cookie_copy)
            except OSError:
                pass

    final = None
    try:
        requested = (info.get("requested_downloads") or [{}])[0]
        final = requested.get("filepath") or info.get("filepath")
    except Exception:
        pass
    if not final or not os.path.exists(final):
        files = [p for p in Path(folder).glob(f"{out_name}.*") if p.is_file()]
        files.sort(key=lambda p: p.stat().st_mtime)
        final = str(files[-1]) if files else None
    if not final:
        raise RuntimeError("The download finished but the file could not be found.")
    return final


PERMANENT_ERROR_WORDS = (
    "private", "login", "log in", "sign in", "members-only", "members only", "age-restricted",
    "confirm your age", "unavailable", "removed", "deleted", "does not exist", "not exist",
    "unsupported url", "no suitable extractor", "copyright", "not available in your country",
    "geo", "404", "is not a valid url", "invalid url", "no video could be found",
    "has been terminated", "no longer available", "video is not available", "post is not available",
)
# Messages like "rate-limit reached or login required" are really temporary (the site is just
# refusing us for a while), so those are retried - with a different method each time.
TEMPORARY_ERROR_WORDS = (
    "rate-limit", "rate limit", "too many requests", "429", "try again", "timed out", "timeout",
    "temporar", "connection", "reset by peer", "network", "502", "503", "504", "500",
    "unable to extract", "unable to download", "http error 403", "403", "empty response", "ssl",
)


HARD_BROKEN_WORDS = (
    "404", "removed", "deleted", "does not exist", "not exist", "unsupported url", "no suitable extractor",
    "copyright", "is not a valid url", "invalid url", "has been terminated", "private video",
    "this video is private", "no video could be found", "no longer available",
)


BOT_CHECK_WORDS = ("not a bot", "confirm you", "bot check", "sign in to confirm")


def is_bot_check(exc):
    """YouTube's 'Sign in to confirm you are not a bot' - it depends on the connection and the method, so retrying works."""
    low = ANSI_RE.sub("", str(exc)).lower().replace("\u2019", "'")
    return any(w in low for w in BOT_CHECK_WORDS)


def is_permanent_error(exc):
    """True only for links that are really broken: removed, private, unsupported, needs a login..."""
    low = ANSI_RE.sub("", str(exc)).lower()
    if is_bot_check(exc):
        return False
    if any(word in low for word in HARD_BROKEN_WORDS):  # clearly gone: never retry
        return True
    if any(word in low for word in TEMPORARY_ERROR_WORDS):  # the site is just refusing us for now
        return False
    return any(word in low for word in PERMANENT_ERROR_WORDS)


MOBILE_UA = (
    "Mozilla/5.0 (iPhone; CPU iPhone OS 17_4 like Mac OS X) AppleWebKit/605.1.15 "
    "(KHTML, like Gecko) Version/17.4 Mobile/15E148 Safari/604.1"
)
# Other front-end sites that show the same public posts. Used only after the normal ways failed.
MIRROR_SITES = {
    "instagram.com": ["ddinstagram.com", "kkinstagram.com"],
    "tiktok.com": ["vxtiktok.com", "tnktok.com"],
    "twitter.com": ["fxtwitter.com", "vxtwitter.com"],
    "x.com": ["fixupx.com", "fxtwitter.com"],
}


def normalize_link(url):
    """Clean a TikTok link: follow short links (vm.tiktok.com/..., tiktok.com/t/...) to the real
    video page and drop the tracking parameters that sharing adds. Other sites are left alone."""
    try:
        from urllib.parse import urlsplit, urlunsplit
        from urllib.request import Request, urlopen

        parts = urlsplit(url)
        host = (parts.hostname or "").lower()
        if not (host == "tiktok.com" or host.endswith(".tiktok.com")):
            return url
        if host.split(".")[0] in ("vm", "vt") or parts.path.startswith("/t/"):
            with urlopen(Request(url, headers={"User-Agent": MOBILE_UA}), timeout=15) as resp:
                parts = urlsplit(resp.geturl())
        path = parts.path.replace("/photo/", "/video/")
        return urlunsplit((parts.scheme or "https", parts.netloc, path, "", ""))
    except Exception:
        return url


def impersonation_opts():
    """Makes the connection look like a real Chrome browser (TikTok and others check this).
    Needs the optional curl_cffi package; without it this step does nothing."""
    try:
        import curl_cffi  # noqa: F401
        from yt_dlp.networking.impersonate import ImpersonateTarget

        return {"impersonate": ImpersonateTarget("chrome")}
    except Exception:
        return {}


def mirror_url(url, round_no):
    m = re.match(r"^(https?://)(?:www\.|m\.|mobile\.)?([^/:?#]+)(.*)$", url, re.I)
    if not m:
        return None
    options = MIRROR_SITES.get(m.group(2).lower())
    if not options:
        return None
    return f"{m.group(1)}{options[round_no % len(options)]}{m.group(3)}"


def retry_strategy(attempt, url, kind):
    """The way to download on this attempt (1 = first try): a different method every time, so a
    link that fails one way is tried through other connections, identities, clients and sites.
    Returns (url_to_use, extra_options, short_description)."""
    yt = lambda *clients: {"extractor_args": {"youtube": {"player_client": list(clients)}}}
    plan = [
        ("normal way", {}),
        ("real-browser connection", impersonation_opts()),
        ("IPv4 connection", {"source_address": "0.0.0.0"}),
        ("mobile browser identity", {"http_headers": {"User-Agent": MOBILE_UA}}),
        ("YouTube Android client", yt("android", "web")),
        ("alternative site", None),
        ("YouTube iOS client + IPv4", dict(yt("ios"), source_address="0.0.0.0")),
        ("simpler quality", {"format": "worstaudio/worst" if kind == "audio" else "best"}),
        ("YouTube TV client", yt("tv", "web_safari")),
    ]
    label, opts = plan[(attempt - 1) % len(plan)]
    if opts is None:
        alt = mirror_url(url, (attempt - 1) // len(plan))
        if alt:
            return alt, {}, "alternative site " + re.sub(r"^https?://([^/]+).*$", r"\1", alt)
        label, opts = plan[0][0], {}
    return url, opts, label


def retry_wait(attempt):
    """Seconds to wait before the next try: 3, 6, 9 ... up to 30."""
    return min(3 * attempt, 30)


def download_links(links_text, fmt, folder, retries=20):
    folder_name, links = parse_input(links_text)
    if not links:
        raise gr.Error("Please paste at least one link first (it must start with http:// or https://).")
    if len(links) > MAX_LINKS:
        raise gr.Error(f"Please paste at most {MAX_LINKS} links at a time (you pasted {len(links)}).")
    base = (folder or "").strip().strip('"')
    if not base:
        raise gr.Error("Please type a download folder path or choose one with Browse.")
    sub = clean_name(folder_name)
    target = Path(base) / sub if sub else Path(base)
    try:
        target.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        raise gr.Error(f"Could not use this folder: {e}")
    try:
        import yt_dlp
    except ImportError:
        raise gr.Error("The downloader is not installed. Close the app and run run.bat again.")

    codec, kind = DL_FORMATS[fmt]
    prefix = "audio" if kind == "audio" else "video"
    start = next_number(target, prefix)
    ffmpeg_loc = ffmpeg_dir()
    total = len(links)
    max_retries = max(0, int(retries or 0))
    done_files = []
    problems = []
    failed_numbers = set()
    log = [f"Starting - {total} link(s)", f"Folder: {target}", "-" * 40, ""]
    _dl_cancel.clear()
    cancelled = False
    stopped_at = total + 1

    def show(live=None):
        return None, "", "\n".join(log + ([live] if live else []))

    yield show()

    for i, url in enumerate(links):
        if _dl_cancel.is_set():
            cancelled, stopped_at = True, i + 1
            break
        log.append(f"[{i + 1}/{total}]  {url}")
        yield show()
        if not re.match(r"^https?://", url, re.I):
            problems.append(f"Link {i + 1}: {url}\n   This is not a valid link (it must start with http:// or https://).")
            failed_numbers.add(i + 1)
            log.append("   X not a valid link")
            log.append("")
            yield show()
            continue

        clean_url = normalize_link(url)
        if clean_url != url:
            log.append(f"   link cleaned: {clean_url}")
            yield show()
        engine_note_shown = False
        attempt = 0
        while True:
            if _dl_cancel.is_set():
                cancelled, stopped_at = True, i + 1
                break
            attempt += 1
            state = {"frac": 0.0}
            result = {}

            use_url, extra, how = retry_strategy(attempt, clean_url, kind)
            if attempt > 1:
                log.append(f"   retry {attempt - 1}: {how}")
                yield show()

            def worker(_url=use_url, _extra=extra, _n=start + i, _state=state, _result=result):
                try:
                    _result["file"] = download_one(
                        yt_dlp, _url, codec, kind, str(target), f"{prefix} {_n}", ffmpeg_loc,
                        lambda f: _state.update(frac=f), _extra,
                    )
                except Exception as err:
                    _result["err"] = err

            th = threading.Thread(target=worker, daemon=True)
            th.start()
            last = -1
            while th.is_alive():
                th.join(0.5)
                if _dl_cancel.is_set():
                    th.join(10)  # the download stops at its next progress step
                    break
                pct = int(state["frac"] * 100)
                if pct != last:
                    last = pct
                    yield show(f"   downloading... {pct}%")
            if _dl_cancel.is_set():
                cancelled, stopped_at = True, i + 1
                remove_partial(target, f"{prefix} {start + i}")
                log.append("   X cancelled by you")
                log.append("")
                yield show()
                break
            try:
                if "err" in result:
                    raise result["err"]
                saved = result["file"]
                done_files.append(saved)
                log.append(f"   -> {os.path.basename(saved)}")
                log.append("")
                yield show()
                break
            except Exception as e:
                # Keep trying (network problems, timeouts, busy servers) until it works.
                # Stop early only for errors that can never be fixed by trying again.
                if is_permanent_error(e) or attempt > max_retries:
                    tries = f" (tried {attempt} time{'s' if attempt != 1 else ''})" if attempt > 1 else ""
                    problems.append(f"Link {i + 1}: {url}{tries}\n   {explain_error(e)}")
                    failed_numbers.add(i + 1)
                    short = " ".join(ANSI_RE.sub("", str(e)).replace("ERROR:", "").split())[:170]
                    log.append(f"   X failed{tries}: {short}")
                    log.append("")
                    yield show()
                    break
                if attempt >= 2 and looks_outdated(e):
                    if not getattr(sys, "frozen", False) and not _ytdlp_update_tried:
                        log.append("   ! the downloader engine looks out of date - updating it now (needs internet, up to a minute)...")
                        yield show()
                        newer = update_ytdlp()
                        if newer:
                            yt_dlp = newer
                            log.append("   engine updated, trying again")
                        else:
                            log.append("   could not update the engine automatically (run update_downloader.bat)")
                    elif getattr(sys, "frozen", False) and not engine_note_shown:
                        engine_note_shown = True
                        log.append("   ! this .exe has an older downloader engine inside; rebuild it after update_downloader.bat for newer sites")
                wait = retry_wait(attempt)
                nxt = retry_strategy(attempt + 1, clean_url, kind)[2]
                log.append(f"   ! failed, next try in {wait}s using: {nxt} (retry {attempt} of {max_retries})")
                yield show()
                _dl_cancel.wait(wait)  # a normal pause, but Cancel ends it at once
        if cancelled:
            break

    store_setting("download_dir", base)

    ok = len(done_files)
    failed_nums = sorted(failed_numbers)
    if cancelled:
        headline = f"Cancelled. {ok}/{total} downloaded."
        if stopped_at <= total:
            left = ", ".join(str(n) for n in range(stopped_at, total + 1))
            headline += f" Not downloaded: link {left}."
        if failed_nums:
            headline += " Failed: link " + ", ".join(str(n) for n in failed_nums) + "."
    elif failed_nums:
        nums = ", ".join(str(n) for n in failed_nums)
        word = "Link" if len(failed_nums) == 1 else "Links"
        headline = f"{ok}/{total} downloaded. {word} {nums} failed."
    else:
        headline = f"{ok}/{total} downloaded. No link failed."
    log.append("-" * 40)
    log.append(headline)
    lines = [headline, "", f"Saved in: {target}"]
    if done_files:
        lines.append("Files: " + ", ".join(os.path.basename(f) for f in done_files))
    if problems:
        lines.append("")
        lines.append("Failed links:\n")
        lines.append("\n\n".join(problems))
    gr.Info(headline)
    yield done_files or None, "\n".join(lines), "\n".join(log)


# ---------- UI ----------
def folder_row(value, label, info):
    folder = gr.Textbox(label=label, value=value, info=info)
    with gr.Row():
        browse_btn = gr.Button("Browse...")
        open_btn = gr.Button("Open folder")
    browse_btn.click(browse_folder, folder, folder)
    open_btn.click(open_folder, folder, None)
    return folder


def save_folder_row():
    """Save-folder controls placed under the Result player."""
    return folder_row(str(load_setting("save_dir", DEFAULT_SAVE_DIR)).replace("Zynth Studio", "Pindi Studio"), "Save folder", "Results are saved here automatically.")


# ---- PINDI STUDIO colours: warm cream + deep teal (main buttons, highlights) + coral (second buttons) ----
TEAL = gr.themes.Color(c50="#effaf8", c100="#d5f2ee", c200="#aae5dd", c300="#74d1c6", c400="#3fb7aa",
                       c500="#1a9a8d", c600="#0f766e", c700="#0b5d56", c800="#094a45", c900="#073b37", c950="#03221f")
CORAL = gr.themes.Color(c50="#fff3ee", c100="#ffe6dc", c200="#ffcdb9", c300="#ffab8e", c400="#ff7a59",
                        c500="#f4592f", c600="#d94421", c700="#b43519", c800="#922d19", c900="#782a1a", c950="#41120a")
SAND = gr.themes.Color(c50="#fbf8f2", c100="#f6f1e7", c200="#ebe2d0", c300="#ded5c3", c400="#b9ae98",
                       c500="#8f866f", c600="#6f6856", c700="#5a5446", c800="#3d392f", c900="#2a2823", c950="#1f2a28")
try:
    SKY_THEME = gr.themes.Soft(
        primary_hue=TEAL, secondary_hue=CORAL, neutral_hue=SAND,
        font=[gr.themes.Font("Roboto"), "Segoe UI", "Arial", "sans-serif"],
        font_mono=[gr.themes.Font("Consolas"), "monospace"],
    ).set(
        # the same light look in light and dark mode
        body_background_fill="#f6f1e7", body_background_fill_dark="#f6f1e7",
        body_text_color="#1f2a28", body_text_color_dark="#1f2a28",
        body_text_color_subdued="#5f6b68", body_text_color_subdued_dark="#5f6b68",
        background_fill_primary="#fffdf8", background_fill_primary_dark="#fffdf8",
        background_fill_secondary="#f6f1e7", background_fill_secondary_dark="#f6f1e7",
        block_background_fill="#fffdf8", block_background_fill_dark="#fffdf8",
        block_border_color="#ded5c3", block_border_color_dark="#ded5c3",
        block_label_background_fill="#f6f1e7", block_label_background_fill_dark="#f6f1e7",
        block_label_text_color="#5f6b68", block_label_text_color_dark="#5f6b68",
        block_title_text_color="#1f2a28", block_title_text_color_dark="#1f2a28",
        border_color_primary="#ded5c3", border_color_primary_dark="#ded5c3",
        input_background_fill="#ffffff", input_background_fill_dark="#ffffff",
        input_border_color="#ded5c3", input_border_color_dark="#ded5c3",
        input_border_color_focus="#0f766e", input_border_color_focus_dark="#0f766e",
        input_placeholder_color="#9aa39f", input_placeholder_color_dark="#9aa39f",
        color_accent_soft="#d5f2ee", color_accent_soft_dark="#d5f2ee",
        slider_color="#0f766e", slider_color_dark="#0f766e",
        checkbox_background_color_selected="#0f766e", checkbox_background_color_selected_dark="#0f766e",
        checkbox_border_color_selected="#0f766e", checkbox_border_color_selected_dark="#0f766e",
        button_primary_background_fill="#0f766e", button_primary_background_fill_hover="#0b5d56",
        button_primary_background_fill_dark="#0f766e", button_primary_background_fill_hover_dark="#0b5d56",
        button_primary_text_color="#ffffff", button_primary_text_color_dark="#ffffff",
        button_secondary_background_fill="#ffe6dc", button_secondary_background_fill_hover="#ffd3c2",
        button_secondary_background_fill_dark="#ffe6dc", button_secondary_background_fill_hover_dark="#ffd3c2",
        button_secondary_text_color="#7a2a12", button_secondary_text_color_dark="#7a2a12",
        button_secondary_border_color="#ff7a59", button_secondary_border_color_dark="#ff7a59",
    )
except Exception:
    SKY_THEME = gr.themes.Soft(primary_hue="teal", secondary_hue="orange", neutral_hue="stone")
try:
    demo = gr.Blocks(title=APP_NAME, theme=SKY_THEME)
except TypeError:  # a Gradio version without this argument
    demo = gr.Blocks(title=APP_NAME)

with demo:
    gr.HTML(
        "<style>html,body{scroll-behavior:auto!important;overscroll-behavior:none}"
        "@import url('https://fonts.googleapis.com/css2?family=Roboto:wght@400;500;700&display=swap');"
        "*{font-family:Roboto,'Segoe UI',Arial,sans-serif}"
        "button.selected,.tab-nav button.selected{border-color:#0f766e!important;color:#0f766e!important}"
        ".gradio-container *{scroll-behavior:auto!important}</style>"
    )
    gr.Markdown(f"# {APP_NAME}\nInworld 360 + Voice Changer + ASMR + Downloader + Chat - free, running on your computer.")

    with gr.Tab("Inworld 360"):
        with gr.Row():
            with gr.Column(scale=3):
                c_text = gr.Textbox(
                    label="Text",
                    placeholder="Paste or type the text for the voiceover here...",
                    lines=16,
                    max_lines=16,
                    max_length=MAX_TTS_CHARS,
                    info=f"Up to {MAX_TTS_CHARS:,} characters.",
                )
                c_btn = gr.Button("Generate", variant="primary")
                c_status = gr.Textbox(label="Status", interactive=False, lines=1)
                c_out = gr.Audio(label="Result", type="filepath")
                c_dl = gr.DownloadButton("Download MP3", visible=False)
                c_aname = gr.Textbox(
                    label="Audio name (write this first)", placeholder="e.g. Audio 1", lines=1, max_lines=1,
                    info="The result is saved with this name. Generate starts only after you write a name.",
                )
                c_folder = save_folder_row()
                c_saved = gr.Textbox(label="Saved at", interactive=False)
            with gr.Column(scale=1, min_width=330):
                c_iw_voice = gr.Dropdown(
                    inworld_voice_list(), value=inworld_voice_id(), label="Inworld voice", allow_custom_value=True,
                    info="The voice for the Inworld 360 model. Pick one, or add your own below. "
                         "The voice you use is remembered.",
                )
                c_iw_new = gr.Textbox(
                    label="Add a voice", placeholder="Exact voice name from Inworld, e.g. Dennis",
                    info="Type the name exactly as in your Inworld Portal, then press Add.",
                )
                with gr.Row():
                    c_iw_add = gr.Button("Add voice")
                    c_iw_del = gr.Button("Remove selected voice")
                # The saved-voice list of the Voice Changer tab (kept hidden here; it is shared with that tab)
                c_saved_dd = gr.Dropdown(voice_choices(), value=NO_SAVED, label="Voice", visible=False)
                c_preview = gr.Audio(label="Voice preview", interactive=False, visible=False)
                c_model = gr.Dropdown(
                    TTS_MODELS, value=MATCH_MODEL, label="Model",
                    info="'Voice Changer match' reads the text with a small light voice, then changes it to your "
                         "sample with the Voice Changer engine - the same match as the Voice Changer tab. "
                         "'Inworld 360' is an online voice (no sample needed).",
                )
                with gr.Group(visible=False) as c_inworld_box:
                    gr.Markdown(
                        "**Inworld 360 settings**\n\n"
                        f"- Model: `{INWORLD_MODEL_ID}`\n"
                        "- Format: **MP3**\n"
                        f"- Delivery: **{INWORLD_DELIVERY.title()}**\n"
                        "- Language: **Auto** (detected from your text)\n"
                        "- Speaking rate: use **Talking speed** below (1.0 = normal)\n\n"
                        "Online voice: your text is sent to Inworld. No voice sample is needed. "
                        f"The API key is read from the `{INWORLD_KEY_VAR}` environment variable on this computer."
                    )
                c_vc = gr.Dropdown(
                    VC_CHOICES, value=list(VC_MODELS)[0], label="Match engine",
                    info="Used by 'Voice Changer match'. Pick the same one that works best for you in the Voice Changer tab.",
                )
                c_clean = gr.Checkbox(
                    value=True, label="Clean background noise from the sample",
                    info="Recommended. The voice copies the noise in the sample, so removing it first gives a clearer voice.",
                )
                c_steps = gr.Slider(
                    10, 40, value=10, step=1,
                    label="Quality steps",
                    info="Only used by Max match or the Seed-VC engine. 10 = fastest. Raise it (20-30) for a cleaner result; it gets slower.",
                )
                c_lang = gr.Dropdown(list(LANGUAGES), value="English", label="Language")
                c_speed = gr.Slider(0.5, 1.5, value=1.0, step=0.05, label="Talking speed")
                c_sample = gr.File(
                    label="Voice sample - any audio or video file (clear speech, 5-30 seconds)",
                    file_count="single",
                    type="filepath",
                )
                c_mic = gr.Audio(
                    label="Or record from microphone",
                    sources=["microphone"],
                    type="filepath",
                )
                c_name = gr.Textbox(label="Save this sample as", placeholder="Voice name, e.g. Ali")
                with gr.Row():
                    c_save_btn = gr.Button("Save voice")
                    c_del_btn = gr.Button("Delete voice")
        c_btn.click(
            clone_voice,
            [c_sample, c_mic, c_saved_dd, c_text, c_lang, c_speed, c_model, c_steps, c_clean, c_vc, c_folder, c_iw_voice, c_aname],
            [c_out, c_saved, c_status],
            show_progress="hidden",
        )
        c_iw_add.click(inworld_add_voice, c_iw_new, [c_iw_voice, c_iw_new], show_progress="hidden")
        c_iw_new.submit(inworld_add_voice, c_iw_new, [c_iw_voice, c_iw_new], show_progress="hidden")
        c_iw_del.click(inworld_remove_voice, c_iw_voice, c_iw_voice, show_progress="hidden")
        c_model.change(
            lambda m: gr.update(visible=(m == INWORLD_MODEL)), c_model, c_inworld_box, show_progress="hidden",
        )
        c_out.change(
            lambda path: gr.update(value=path, visible=True) if path and str(path).lower().endswith(".mp3")
            else gr.update(value=None, visible=False),
            c_out, c_dl, show_progress="hidden",
        )

    with gr.Tab("Voice Changer"):
        gr.HTML(
            "<style>.vc-row{border:1px solid var(--border-color-primary,#444)!important;border-top:none!important;gap:0!important;align-items:stretch!important}"
            ".vc-head{border:1px solid var(--border-color-primary,#444)!important;gap:0!important;margin-top:6px}"
            ".vc-row>div,.vc-head>div{border-right:1px solid var(--border-color-primary,#444);min-width:0!important;display:flex;align-items:center;justify-content:center}"
            ".vc-row>div:last-child,.vc-head>div:last-child{border-right:none}"
            ".vc-row .block,.vc-head .block{border:none!important;box-shadow:none!important;background:transparent!important}"
            "</style>"
            "<div style='opacity:.8;font-size:13px'>Drop an original audio in each row you need (up to 6), choose a saved target voice for it, "
            "then press <b>Change Voice</b>. Rows are converted one by one.</div>"
        )
        v_overall = gr.HTML(vc_overall(0))
        with gr.Row(elem_classes="vc-head"):
            for title, sc in (("SrNo.", 1), ("Source Audio", 4), ("Target Audio", 3), ("Progress", 2), ("Status", 2)):
                with gr.Column(scale=sc, min_width=0):
                    gr.HTML(f"<div style='text-align:center;font-weight:700;padding:10px 4px;font-size:15px;width:100%'>{title}</div>")
        v_row_files, v_row_voices, v_row_prog, v_row_stat, v_row_stop = [], [], [], [], []
        for r in range(1, VC_ROWS + 1):
            with gr.Row(elem_classes="vc-row"):
                with gr.Column(scale=1, min_width=0):
                    gr.HTML(f"<div style='text-align:center;font-size:15px;width:100%'>{r}</div>")
                v_row_files.append(gr.File(show_label=False, file_count="single", type="filepath", height=96, scale=4,
                                           min_width=0, container=False))
                v_row_voices.append(gr.Dropdown(list_voices(), value=None, show_label=False, container=False, scale=3,
                                                min_width=0))
                with gr.Column(scale=2, min_width=0):
                    v_row_prog.append(gr.HTML(vc_progress_cell(None)))
                with gr.Column(scale=2, min_width=0):
                    v_row_stat.append(gr.HTML(vc_status_cell("")))
                    v_row_stop.append(gr.Button("Stop", variant="stop", size="sm", min_width=60))
        gr.HTML("<div style='font-size:12px;opacity:.7;margin-top:6px'>When a row reaches 100%, the changed voice is saved in the "
                "chosen folder with the same name as the original audio (e.g. \"audio 1\"). If that name already exists, "
                "\" changed voice\" is added to the name.</div>")
        with gr.Row():
            v_model = gr.Dropdown(
                VC_CHOICES,
                value=VC_DEFAULT,
                label="Voice changer model",
                info="Each model downloads once, the first time you use it.",
                scale=3,
            )
            v_steps = gr.Slider(
                10, 50, value=10, step=1,
                label="Seed-VC quality steps",
                info="Only used by Seed-VC. 10 = fastest. 20-30 = better quality, slower.",
                scale=3,
            )
            v_clean = gr.Checkbox(
                value=True, label="Clean background noise",
                info="Recommended. Cleans the original and the sample, and polishes the result.",
                scale=2,
            )
        v_folder = save_folder_row()
        v_btn = gr.Button("Change Voice", variant="primary")
        with gr.Row():
            v_out = gr.Audio(label="Latest result", type="filepath")
            v_files = gr.File(label="All converted files", file_count="multiple", interactive=False)
        v_saved = gr.Textbox(label="Details", interactive=False, lines=4, max_lines=12)

        with gr.Accordion("Add or manage target voices (saved voices appear in every row's Target Audio list)", open=False):
            v_saved_dd = gr.Dropdown(
                voice_choices(), value=NO_SAVED, label="Saved voices",
                info="Select one to preview it or to delete it.",
            )
            v_preview = gr.Audio(label="Saved voice preview", interactive=False)
            v_target = gr.File(
                label="New target voice sample - any audio or video file (5-30 seconds; a longer file is cut to 30 seconds)",
                file_count="single",
                type="filepath",
            )
            v_target_mic = gr.Audio(label="Or record the target voice from microphone", sources=["microphone"], type="filepath")
            with gr.Row():
                v_name = gr.Textbox(label="Save this target as", placeholder="Voice name, e.g. Jason", scale=3)
                v_save_btn = gr.Button("Save voice", scale=1)
                v_del_btn = gr.Button("Delete saved voice", scale=1)

        v_btn.click(
            change_voice,
            [*v_row_files, *v_row_voices, v_model, v_steps, v_clean, v_folder],
            [v_out, v_saved, v_files, v_overall, *v_row_prog, *v_row_stat],
            show_progress="hidden",
        )

        for r, stop_btn in enumerate(v_row_stop):
            stop_btn.click(make_stop_handler(r), None, None, queue=False, show_progress="hidden")

        def refresh_row_voices(*current):
            names = list_voices()
            return [gr.update(choices=names, value=(c if c in names else None)) for c in current]


    # Saved voices are shared by both tabs, so every change updates both lists.
    c_save_btn.click(save_voice, [c_name, c_sample, c_mic], [c_saved_dd, v_saved_dd]).then(refresh_row_voices, v_row_voices, v_row_voices)
    v_save_btn.click(save_voice, [v_name, v_target, v_target_mic], [c_saved_dd, v_saved_dd]).then(refresh_row_voices, v_row_voices, v_row_voices)
    c_del_btn.click(delete_voice, c_saved_dd, [c_saved_dd, v_saved_dd]).then(refresh_row_voices, v_row_voices, v_row_voices)
    v_del_btn.click(delete_voice, v_saved_dd, [c_saved_dd, v_saved_dd]).then(refresh_row_voices, v_row_voices, v_row_voices)
    c_saved_dd.change(preview_voice, c_saved_dd, c_preview)
    v_saved_dd.change(preview_voice, v_saved_dd, v_preview)

    with gr.Tab("ASMR"):
        gr.Markdown(
            "**Remove the background song and keep the rest.** Add audio or video files (up to 6). The voiceover and the "
            "ASMR sounds stay, the song is removed. A video keeps its picture and gets the clean sound; an audio file "
            "becomes a clean WAV. Files are processed one by one and saved in the chosen folder with the same name "
            "(if that name exists, ' no music' is added)."
        )
        a_mode = gr.Radio(
            ["Separate everything into different files (voiceover, singing, drums, bass, instruments, ASMR sounds)",
             "Voiceover only (removes all music - and also the ASMR sounds)",
             "Voiceover + bright ASMR sounds (crinkle, brushing, tapping) - strong music removal",
             "Voiceover + all ASMR sounds - weaker music removal"],
            value="Voiceover + bright ASMR sounds (crinkle, brushing, tapping) - strong music removal",
            label="What do you want?",
            info="The first option gives every sound as its own file. The others make ONE clean file. "
                 "If music is still audible, choose 'Voiceover only': it removes the most music.",
        )
        a_files = gr.File(label="Audio or video files (up to 6)", file_count="multiple", type="filepath")
        a_strength = gr.Radio(
            ["Normal (1 pass)", "Strong (2 passes, slower) - recommended for light background music",
             "Maximum (best model + 2 passes, slowest, downloads about 330 MB once)"],
            value="Strong (2 passes, slower) - recommended for light background music",
            label="Removal strength",
        )
        a_lyrics = gr.Radio(
            ["Off", "Normal - remove quiet singing behind the voiceover", "Strong - remove it more"],
            value="Normal - remove quiet singing behind the voiceover",
            label="Song lyrics (singing voice)",
            info="Removes the song's singing when the voiceover is louder than it. It learns the singing from the "
                 "pauses of the voiceover, so it works best when the voiceover has pauses.",
        )
        a_keep = gr.Slider(
            0, 100, value=0, step=5,
            label="Let some of the removed sound back in (%)",
            info="0 = remove as much as possible. If some ASMR sound is missing, raise this a little (for example 20-40); "
                 "a little of the song comes back with it.",
        )
        a_folder = save_folder_row()
        a_debug = gr.Checkbox(
            value=False, label="Also save the two separated parts for checking (voice part and other part)",
            info="Use this if music is still audible: listen to the 'voice part' file. If the music is in it, the engine "
                 "mixed the music into the voice.",
        )
        a_btn = gr.Button("Remove music", variant="primary")
        a_status = gr.Textbox(label="Progress", interactive=False, lines=8, max_lines=18)
        a_audio = gr.Audio(label="Latest audio result", type="filepath", interactive=False)
        gr.Markdown(
            "First use downloads a small model (about 80 MB) and the music remover (Demucs) must be installed once "
            "with **install_demucs.bat**. ASMR sounds that are soft and tonal (humming, singing bowls) can be treated "
            "as music, and a strong drum beat can leave small clicks, so listen to the result."
        )
        a_btn.click(asmr_remove_music, [a_files, a_mode, a_strength, a_lyrics, a_keep, a_folder, a_debug], [a_status, a_audio], show_progress="hidden")

    with gr.Tab("Downloader"):
        gr.Markdown(
            "Paste links from **YouTube, TikTok, Instagram** (and 1000+ other sites), one per line, "
            f"up to {MAX_LINKS} at a time. Files are named **audio 1, audio 2...** (or **video 1, video 2...**) "
            "in the same order as your links. To save them in their own folder, type a folder name on the first "
            "line, leave one empty line, then paste the links. Only download content you own or have permission to use."
        )
        d_urls = gr.Textbox(
            label=f"Links (one per line, up to {MAX_LINKS})",
            placeholder="My folder name (optional)\n\nhttps://...\nhttps://...\nhttps://...",
            lines=14,
            max_lines=40,
        )
        with gr.Row():
            d_fmt = gr.Radio(list(DL_FORMATS), value="Audio - MP3", label="Download as", scale=3)
            d_btn = gr.Button("Start download", variant="primary", scale=1)
            d_clean = gr.Button("Clean", scale=0, min_width=90)
            d_cancel = gr.Button("Cancel", variant="stop", size="sm", scale=0, min_width=90)
        d_retries = gr.Slider(
            0, 50, value=20, step=1,
            label="Retry attempts if a download fails",
            info="If a link fails, the app keeps trying again - each time a different way (other connection, "
                 "browser identity, app client, even other sites). Links that are really broken "
                 "(removed, private, needs login) are not retried.",
        )
        with gr.Accordion("Cookies (optional) - for Instagram / TikTok / YouTube posts that need a login", open=False):
            gr.Markdown(
                "Public links work without this. If a link says it needs a login, export **your own** cookies.txt from your "
                "browser (while logged in to that site) with a browser extension such as *Get cookies.txt LOCALLY*, pick the "
                "site and upload the file here. It is kept only on this PC and used only for downloads from that site. "
                "Cookies work like a saved login, so never share them with anyone."
            )
            with gr.Row():
                d_ck_site = gr.Dropdown(["Instagram", "TikTok", "YouTube", "Facebook", "Twitter / X", "All sites"],
                                        value="Instagram", label="Site", scale=1)
                d_ck_file = gr.File(label="cookies.txt file", file_count="single", type="filepath", scale=2)
            with gr.Row():
                d_ck_save = gr.Button("Save cookies")
                d_ck_clear = gr.Button("Remove all saved cookies")
            d_ck_msg = gr.Markdown("")
            d_ck_save.click(save_cookie_file, [d_ck_site, d_ck_file], d_ck_msg, show_progress="hidden")
            d_ck_clear.click(remove_cookie_files, None, d_ck_msg, show_progress="hidden")
        d_folder = folder_row(
            str(load_setting("download_dir", DEFAULT_DOWNLOAD_DIR)).replace("Zynth Studio", "Pindi Studio"),
            "Download folder",
            "Downloads are saved here.",
        )
        d_log = gr.Textbox(
            label="Progress",
            placeholder="Each link is shown here as it downloads, for example [1/6] ... -> audio_1.mp3",
            lines=8,
            max_lines=14,
            interactive=False,
        )
        d_files = gr.File(label="Downloaded files", file_count="multiple", interactive=False)
        d_comments = gr.Textbox(
            label="Comments",
            placeholder="When finished, this shows how many links were downloaded (for example 4/4) and which link number failed, with the reason.",
            lines=6,
            interactive=False,
        )
        gr.HTML(
            """
<style>
@keyframes zthanks_shift { 0% { background-position: 0% 50%; } 100% { background-position: 200% 50%; } }
.zthanks {
  text-align: center; font-size: 30px; font-weight: 800; letter-spacing: .5px; margin: 28px 0 10px;
  background: linear-gradient(90deg, #0f766e, #14b8a6, #ff7a59, #0f766e);
  background-size: 200% auto; -webkit-background-clip: text; background-clip: text;
  -webkit-text-fill-color: transparent; color: transparent;
}
.zthanks:hover { animation: zthanks_shift 4s linear infinite; }  /* animated only on hover: a running animation made scrolling laggy */
</style>
<div class="zthanks">Thanks For Choosing Our Services</div>
"""
        )
        d_cancel.click(cancel_download, None, None, queue=False, show_progress="hidden")
        d_clean.click(clean_links, d_urls, d_urls, show_progress="hidden")
        d_btn.click(download_links, [d_urls, d_fmt, d_folder, d_retries], [d_files, d_comments, d_log], show_progress="hidden")

    def default_chat_address():
        """A friend's app joins the admin's chat by itself when a file chat_address.txt (one line: the chat
        address) sits next to the app, in its 'updates' folder, or in the .zynth_studio folder."""
        for folder in (BASE_DIR, BASE_DIR / "updates", Path.home() / ".zynth_studio"):
            try:
                text = (folder / "chat_address.txt").read_text(encoding="utf-8-sig").strip()
            except Exception:
                continue
            m = re.match(r"^https?://[^/\s]+", text.splitlines()[0].strip() if text else "", re.I)
            if m:
                return m.group(0)
        return ""

    CHAT_DEFAULT_ADDR = default_chat_address()
    chat_role_now = load_setting("chat_role", "Friend (I join)" if CHAT_DEFAULT_ADDR else "Admin (I host the chat)")

    with gr.Tab("Chat"):
        gr.Markdown(
            "**Private realtime chat.** Only accounts created by the **admin** can sign in - there is no sign-up. "
            "The admin starts the chat server on his PC, adds friends (username + password) inside the chat, "
            "and shares the address. Friends choose **Friend** below, paste that address and sign in."
        )
        ch_role = gr.Radio(
            ["Admin (I host the chat)", "Friend (I join)"],
            value=chat_role_now,
            label="I am the",
        )
        with gr.Group(visible=(chat_role_now.startswith("Admin"))) as ch_admin_box:
            gr.Markdown(
                "**Step 1** - set the admin name and password (only needed once; you can change them any time). "
                "**Step 2** - start the server. **Step 3** - sign in below with the admin account, open the "
                "gear button and add your friend's username and password."
            )
            with gr.Row():
                ch_aname = gr.Textbox(label="Admin username", value=load_setting("chat_admin_name", ""), scale=1)
                ch_apass = gr.Textbox(label="Admin password (6+ characters)", type="password", scale=1)
                ch_asave = gr.Button("Save admin account", scale=0, min_width=170)
            with gr.Row():
                ch_start = gr.Button("Start chat server", variant="primary")
                ch_stop = gr.Button("Stop server")
                ch_tunnel = gr.Button("Create internet link (for friends on another network)")
            with gr.Row():
                ch_fixed = gr.Textbox(
                    label="Address for friends (optional - leave empty to use the internet link above)",
                    placeholder="https://my-chat.example.com", scale=4,
                )
                ch_savelink = gr.Button("Save address for friends", scale=1, min_width=200)
            ch_linkmsg = gr.Markdown("")
            ch_status = gr.Markdown("Server is not running.")
        with gr.Group(visible=(not chat_role_now.startswith("Admin"))) as ch_friend_box:
            gr.Markdown(
                "Just **sign in** below with the username and password the admin gave you."
                if CHAT_DEFAULT_ADDR else
                "Paste the chat address the admin gave you, press Connect, then sign in."
            )
            with gr.Accordion("Server address (only change this if the admin tells you to)", open=not CHAT_DEFAULT_ADDR):
                with gr.Row():
                    ch_addr = gr.Textbox(
                        label="Chat address from the admin",
                        placeholder="http://192.168.1.5:8765   or   https://something.trycloudflare.com",
                        value=CHAT_DEFAULT_ADDR or load_setting("chat_address", ""),
                        scale=4,
                    )
                    ch_connect = gr.Button("Connect", variant="primary", scale=1)
            ch_fstatus = gr.Markdown("")
        ch_frame = gr.HTML("")

        def _chat_frame(url):
            safe = url.replace('"', "")
            return (
                f'<iframe src="{safe}" style="width:100%;height:700px;border:1px solid #ded5c3;border-radius:12px;background:#f6f1e7" '
                'allow="clipboard-write"></iframe>'
                f'<div style="margin-top:6px;font-size:12px;opacity:.7">Not loading? Open <a href="{safe}" target="_blank">{safe}</a> in your browser.</div>'
            )

        def _chat_addresses():
            if not chat_server.running:
                return "Server is not running."
            port = chat_server.port
            ip = pindi_chat.lan_ip()
            text = f"Server is **running**.\n\n- On this PC: `http://127.0.0.1:{port}`\n"
            if ip:
                text += f"- Friends on the **same Wi-Fi / network**: `http://{ip}:{port}` (Windows may ask to allow access - click **Allow**)\n"
            if pindi_chat._tunnel.get("url"):
                text += f"- Friends on **any network (internet)**: `{pindi_chat._tunnel['url']}`\n"
            text += "\nKeep Pindi Studio open - the chat works only while it is running on this PC."
            return text

        def chat_pick_role(role):
            store_setting("chat_role", role)
            admin = role.startswith("Admin")
            return gr.update(visible=admin), gr.update(visible=not admin)

        def chat_save_admin(name, pw):
            try:
                chat_server.set_admin(name, pw)
            except pindi_chat.ChatError as e:
                return f"**{e.message}**"
            store_setting("chat_admin_name", name.strip())
            return "Admin account saved. Now press **Start chat server**."

        def chat_start_server():
            if not chat_server.has_admin():
                return "**First set the admin username and password, then press Save admin account.**", ""
            try:
                port = chat_server.start()
            except pindi_chat.ChatError as e:
                return f"**{e.message}**", ""
            return _chat_addresses(), _chat_frame(f"http://127.0.0.1:{port}/")

        def chat_stop_server():
            pindi_chat.stop_tunnel()
            chat_server.stop()
            return "Server stopped.", ""

        def chat_make_tunnel():
            if not chat_server.running:
                yield "**Start the chat server first.**"
                return
            box = {"msg": "Starting..."}
            result = {}

            def work():
                try:
                    result["url"] = pindi_chat.start_tunnel(
                        chat_server.port, BASE_DIR, Path.home() / ".zynth_studio", lambda m: box.update(msg=m)
                    )
                except pindi_chat.ChatError as e:
                    result["err"] = e.message
                except Exception as e:
                    result["err"] = str(e)

            t = threading.Thread(target=work, daemon=True)
            t.start()
            while t.is_alive():
                yield box["msg"]
                time.sleep(1)
            if "err" in result:
                yield f"**{result['err']}**\n\n" + _chat_addresses()
            else:
                yield _chat_addresses()

        def chat_connect_server(addr):
            addr = (addr or "").strip().rstrip("/")
            if not addr:
                return "**Paste the address the admin gave you.**", ""
            if not re.match(r"^https?://", addr, re.I):
                addr = "http://" + addr
            addr = re.match(r"^https?://[^/\s]+", addr, re.I).group(0)
            try:
                import urllib.request

                with urllib.request.urlopen(addr + "/api/ping", timeout=8) as r:
                    ok = b"Chat" in r.read(200)
            except Exception:
                ok = False
            if not ok:
                return (
                    "**Could not reach the chat.** Check the address, that the admin's Pindi Studio is open with the "
                    "server started, and that you are on the same Wi-Fi (or use the internet link).",
                    "",
                )
            store_setting("chat_address", addr)
            return "Connected. Sign in below.", _chat_frame(addr + "/")

        def chat_save_link(fixed):
            addr = (fixed or "").strip() or (pindi_chat._tunnel.get("url") or "")
            m = re.match(r"^https?://[^/\s]+", addr, re.I)
            if not m:
                return "**No address yet.** Press 'Create internet link' first, or type a fixed address."
            addr = m.group(0)
            written = None
            for folder in (BASE_DIR, Path.home() / ".zynth_studio"):
                try:
                    folder.mkdir(parents=True, exist_ok=True)
                    (folder / "chat_address.txt").write_text(addr + "\n", encoding="utf-8")
                    written = folder / "chat_address.txt"
                    break
                except Exception:
                    continue
            if written is None:
                return "**Could not save the file.** Send the address to your friends by message instead."
            store_setting("chat_role", "Admin (I host the chat)")
            return (f"Saved `{addr}` in `{written}`.\n\nGive this **chat_address.txt** file to your friend: "
                    "he puts it next to PindiStudio.exe (or in an 'updates' folder there). When he opens the Chat tab "
                    "he only has to sign in. Note: a free internet link changes every time you create it - "
                    "then save it again and send the new file.")

        ch_savelink.click(chat_save_link, ch_fixed, ch_linkmsg, show_progress="hidden")
        ch_role.change(chat_pick_role, ch_role, [ch_admin_box, ch_friend_box], show_progress="hidden")
        ch_asave.click(chat_save_admin, [ch_aname, ch_apass], ch_status, show_progress="hidden")
        ch_start.click(chat_start_server, None, [ch_status, ch_frame], show_progress="hidden")
        ch_stop.click(chat_stop_server, None, [ch_status, ch_frame], show_progress="hidden")
        ch_tunnel.click(chat_make_tunnel, None, ch_status, show_progress="hidden")
        ch_connect.click(chat_connect_server, ch_addr, [ch_fstatus, ch_frame], show_progress="hidden")

        def chat_autoload():
            # A friend who connected before sees the login page straight away
            addr = CHAT_DEFAULT_ADDR or load_setting("chat_address", "")
            if load_setting("chat_role", chat_role_now).startswith("Friend") and addr:
                return chat_connect_server(addr)
            return gr.update(), gr.update()

        demo.load(chat_autoload, None, [ch_fstatus, ch_frame], show_progress="hidden")

    with gr.Tab("Settings"):
        # Zoom works like the zoom of a browser: Ctrl + / Ctrl - / Ctrl 0 and Ctrl + mouse wheel also work anywhere
        # in the app, and the chosen zoom is remembered for the next start.
        ZOOM_JS = """
(v) => {
  var MIN = 50, MAX = 200, STEP = 10, W = window;
  if (!W.__pindiZoom) {
    var cur = 100, timer = null;
    var clamp = function (x) { x = Math.round(Number(x)); if (!isFinite(x)) x = 100; return Math.max(MIN, Math.min(MAX, x)); };
    W.pindiZoomApply = function (x) {
      cur = clamp(x);
      document.documentElement.style.zoom = String(cur / 100);
      try { localStorage.setItem('pindi_zoom', String(cur)); } catch (e) {}
      return cur;
    };
    W.pindiZoomSync = function (x) {  /* show the value in the Settings slider (that also saves it) */
      var el = document.querySelector('#pindi_zoom input[type="number"]');
      if (el && String(el.value) !== String(x)) { el.value = x; el.dispatchEvent(new Event('input', {bubbles: true})); }
    };
    var change = function (dir) {
      var next = dir === 0 ? 100 : clamp(cur + dir * STEP);
      W.pindiZoomApply(next);
      clearTimeout(timer);
      timer = setTimeout(function () { W.pindiZoomSync(next); }, 350);
    };
    document.addEventListener('keydown', function (e) {
      if (!(e.ctrlKey || e.metaKey) || e.altKey) return;
      var d = null;
      if (e.key === '+' || e.key === '=' || e.code === 'NumpadAdd') d = 1;
      else if (e.key === '-' || e.key === '_' || e.code === 'NumpadSubtract') d = -1;
      else if (e.key === '0' || e.code === 'Numpad0') d = 0;
      if (d === null) return;
      e.preventDefault(); change(d);
    }, true);
    document.addEventListener('wheel', function (e) {
      if (!e.ctrlKey) return;
      e.preventDefault(); change(e.deltaY < 0 ? 1 : -1);
    }, {passive: false, capture: true});
    W.__pindiZoom = true;
  }
  var saved = null;
  try { saved = localStorage.getItem('pindi_zoom'); } catch (e) {}
  var use = W.pindiZoomApply(saved || v || 100);
  if (String(use) !== String(v)) W.pindiZoomSync(use);
}
"""
        APPLY_JS = "(v) => { if (window.pindiZoomApply) { window.pindiZoomApply(v); } return v; }"

        def zoom_save(v):
            store_setting("ui_zoom", int(max(50, min(200, round(float(v or 100))))))

        gr.Markdown(
            "### Zoom\n"
            "Make everything in the app bigger or smaller, like the zoom of a browser. "
            "Shortcuts work everywhere in the app: **Ctrl +** (zoom in), **Ctrl -** (zoom out), "
            "**Ctrl 0** (back to 100%) and **Ctrl + mouse wheel**."
        )
        zoom_slider = gr.Slider(50, 200, value=int(load_setting("ui_zoom", 100)), step=10,
                                label="Zoom (%)", elem_id="pindi_zoom")
        with gr.Row():
            zoom_out = gr.Button("Zoom out  (-)")
            zoom_reset = gr.Button("Reset (100%)")
            zoom_in = gr.Button("Zoom in  (+)")
        zoom_slider.change(zoom_save, zoom_slider, None, js=APPLY_JS, show_progress="hidden")
        zoom_out.click(lambda v: max(50, int(v) - 10), zoom_slider, zoom_slider, show_progress="hidden")
        zoom_in.click(lambda v: min(200, int(v) + 10), zoom_slider, zoom_slider, show_progress="hidden")
        zoom_reset.click(lambda: 100, None, zoom_slider, show_progress="hidden")
        demo.load(None, zoom_slider, None, js=ZOOM_JS, show_progress="hidden")

    gr.Markdown(
        "Note: the XTTS-v2 model license allows **non-commercial** use only. "
        "Without a GPU it runs on CPU, so generation will be slower. "
        "Only use voices you own or have permission to use."
    )


def start_server():
    """Start the Gradio server in the background and return its local URL."""
    def _go():
        return demo.launch(
            server_name="127.0.0.1",
            inbrowser=False,
            prevent_thread_lock=True,
            quiet=True,
        )
    try:
        _, local_url, _ = _go()
    except Exception:
        # custom theme failed -> fall back to the plain theme so the app still opens
        import traceback
        traceback.print_exc()
        try:
            demo.theme = gr.themes.Soft(primary_hue="purple", secondary_hue="lime", neutral_hue="stone")
        except Exception:
            pass
        _, local_url, _ = _go()
    return local_url

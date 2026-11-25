"""AI Studio 01 — Brain & Voice (Kaggle T4 x2, Python 3.13, zero paid APIs)."""
import os, gc, re, json, glob, shutil, asyncio, traceback, subprocess, datetime, zipfile, threading
import warnings
warnings.filterwarnings("ignore")

# Persistent model cache inside the shared Master Hub (must be set before transformers is imported)
_HUB_BASE = "/kaggle/working/AI_Studio_Master_Hub" if os.path.isdir("/kaggle/working") else os.path.abspath("AI_Studio_Master_Hub")
os.environ["HF_HOME"] = os.path.join(_HUB_BASE, "99_model_cache")
os.makedirs(os.environ["HF_HOME"], exist_ok=True)
try:
    from transformers import logging as _hf_logging
    _hf_logging.set_verbosity_error()
except Exception:
    pass

import numpy as np
import soundfile as sf
import gradio as gr

try:
    import torch
except Exception:  # pragma: no cover
    torch = None

# ------------------------------------------------------------------ paths
HUB = _HUB_BASE
SUBFOLDERS = {
    "faces": "00_reference_faces", "scripts": "01_scripts", "voices": "02_voices",
    "images": "03_images", "raw_videos": "04_raw_videos", "music": "05_music",
    "captions": "06_captions_hashtags", "exports": "07_final_exports", "cache": "99_model_cache",
}
for _f in SUBFOLDERS.values():
    os.makedirs(os.path.join(HUB, _f), exist_ok=True)

def _ts():
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

def _err():
    return f"❌ Error Details:\n{traceback.format_exc()}"

# ------------------------------------------------------------------ Google Drive sync
DRIVE = {"gauth": None, "drive": None, "root_id": None, "ids": {}}
SECRETS_PATH = "/kaggle/working/client_secrets.json" if os.path.isdir("/kaggle/working") else "client_secrets.json"

def _find_secrets():
    if os.path.exists(SECRETS_PATH):
        return SECRETS_PATH
    for p in glob.glob("/kaggle/input/**/client_secrets*.json", recursive=True):
        shutil.copy(p, SECRETS_PATH)
        return SECRETS_PATH
    return None

def drive_status():
    return "✅ Google Drive Sync: LIVE CONNECTED (AI_Studio_Master_Hub)" if DRIVE["drive"] else "⚪ Google Drive Sync: Not connected (files are saved locally)"

def drive_step1(upload):
    try:
        from pydrive2.auth import GoogleAuth
        if upload is not None:
            shutil.copy(upload if isinstance(upload, str) else upload.name, SECRETS_PATH)
        path = _find_secrets()
        if not path:
            return "❌ Upload client_secrets.json (or attach it as a Kaggle input) first."
        gauth = GoogleAuth(settings={
            "client_config_backend": "file", "client_config_file": path,
            "oauth_scope": ["https://www.googleapis.com/auth/drive"],
            "get_refresh_token": True,
        })
        gauth.GetFlow()
        gauth.flow.redirect_uri = "urn:ietf:wg:oauth:2.0:oob"
        url = gauth.GetAuthUrl()
        DRIVE["gauth"] = gauth
        return f"🔗 [Click here to sign in with Google]({url})\n\nAfter allowing access, copy the code and paste it in Step 2.\n\n`{url}`"
    except Exception:
        return _err()

def _get_or_create(drive, title, parent=None):
    q = f"title='{title}' and mimeType='application/vnd.google-apps.folder' and trashed=false"
    q += f" and '{parent}' in parents" if parent else " and 'root' in parents"
    found = drive.ListFile({"q": q}).GetList()
    if found:
        return found[0]["id"]
    meta = {"title": title, "mimeType": "application/vnd.google-apps.folder"}
    if parent:
        meta["parents"] = [{"id": parent}]
    f = drive.CreateFile(meta); f.Upload()
    return f["id"]

def drive_step2(code):
    try:
        from pydrive2.drive import GoogleDrive
        if DRIVE["gauth"] is None:
            return "❌ Run Step 1 first.", drive_status()
        DRIVE["gauth"].Auth((code or "").strip())
        drive = GoogleDrive(DRIVE["gauth"])
        root = _get_or_create(drive, "AI_Studio_Master_Hub")
        DRIVE.update(drive=drive, root_id=root, ids={k: _get_or_create(drive, v, root) for k, v in SUBFOLDERS.items()})
        n = 0
        for k, v in SUBFOLDERS.items():
            if k == "cache":
                continue
            for p in glob.glob(os.path.join(HUB, v, "*")):
                if os.path.isfile(p):
                    _upload(p, k); n += 1
        restored = _restore_cache_from_drive()
        return (f"✅ Connected. Hub + 9 subfolders ready on Drive. Synced {n} existing files.\n{restored}"), drive_status()
    except Exception:
        return _err(), drive_status()

def _list_children(parent):
    return DRIVE["drive"].ListFile({"q": f"'{parent}' in parents and trashed=false"}).GetList()

def _restore_cache_from_drive():
    """Download the model cache saved on Drive in an earlier session (skips files already present)."""
    local = os.path.join(HUB, SUBFOLDERS["cache"])
    n = 0
    def walk(fid, dest):
        nonlocal n
        os.makedirs(dest, exist_ok=True)
        for f in _list_children(fid):
            target = os.path.join(dest, f["title"])
            if f["mimeType"] == "application/vnd.google-apps.folder":
                walk(f["id"], target)
            elif not os.path.exists(target) or os.path.getsize(target) != int(f.get("fileSize", -1)):
                f.GetContentFile(target); n += 1
    try:
        walk(DRIVE["ids"]["cache"], local)
        return f"♻️ Model cache restored from Drive: {n} files downloaded (0 = nothing new)."
    except Exception as e:
        return f"⚠️ Could not restore model cache from Drive: {e}"

CACHE_SYNC = {"running": False, "msg": "Model cache not synced yet."}

def _sync_cache_to_drive():
    """Mirror 99_model_cache to Drive (folder structure kept, existing same-size files skipped)."""
    if not DRIVE["drive"] or CACHE_SYNC["running"]:
        return
    CACHE_SYNC["running"] = True
    try:
        local = os.path.join(HUB, SUBFOLDERS["cache"]); n = 0
        def walk(src, fid):
            nonlocal n
            existing = {f["title"]: f for f in _list_children(fid)}
            for name in sorted(os.listdir(src)):
                p = os.path.join(src, name)
                if os.path.islink(p):
                    p = os.path.realpath(p)
                if os.path.isdir(p):
                    sub = existing[name]["id"] if name in existing else _get_or_create(DRIVE["drive"], name, fid)
                    walk(p, sub)
                elif os.path.isfile(p) and not name.endswith(".lock"):
                    if name in existing and int(existing[name].get("fileSize", -1)) == os.path.getsize(p):
                        continue
                    f = DRIVE["drive"].CreateFile({"title": name, "parents": [{"id": fid}]})
                    f.SetContentFile(p); f.Upload(); n += 1
                    CACHE_SYNC["msg"] = f"☁️ Syncing model cache… {n} files uploaded"
        walk(local, DRIVE["ids"]["cache"])
        CACHE_SYNC["msg"] = f"✅ Model cache synced to Drive ({n} new files)."
    except Exception as e:
        CACHE_SYNC["msg"] = f"⚠️ Model cache sync failed: {e}"
    finally:
        CACHE_SYNC["running"] = False

ALL_MODELS = ["NousResearch/Hermes-3-Llama-3.1-8B", "Qwen/Qwen2.5-7B-Instruct", "facebook/musicgen-medium",
              "suno/bark-small", "cvssp/audioldm2"]

def download_all_models():
    try:
        from huggingface_hub import snapshot_download
        lines = []
        for m in ALL_MODELS:
            snapshot_download(m, allow_patterns=["*.json", "*.safetensors", "*.model", "*.txt", "*.py", "*.npz", "*.tiktoken", "tokenizer*"])
            lines.append(f"✅ {m}")
        if DRIVE["drive"]:
            threading.Thread(target=_sync_cache_to_drive, daemon=True).start()
            lines.append("☁️ Uploading the model cache to Google Drive in the background (large: ~60 GB, can take a long time). Press 'Check Cache Sync' to see progress.")
        else:
            lines.append("ℹ️ Connect Google Drive to keep this cache for future sessions.")
        return "\n".join(lines)
    except Exception:
        return _err()

def cache_sync_status():
    return CACHE_SYNC["msg"]

def _upload(path, key):
    f = DRIVE["drive"].CreateFile({"title": os.path.basename(path), "parents": [{"id": DRIVE["ids"][key]}]})
    f.SetContentFile(path); f.Upload()

def save_and_sync(local_path, subfolder_key):
    """Ensure file lives in the hub subfolder and upload it to Drive if connected."""
    dest_dir = os.path.join(HUB, SUBFOLDERS[subfolder_key])
    if os.path.dirname(os.path.abspath(local_path)) != os.path.abspath(dest_dir):
        dest = os.path.join(dest_dir, os.path.basename(local_path))
        shutil.copy(local_path, dest); local_path = dest
    note = "💾 Saved locally"
    if DRIVE["drive"]:
        try:
            _upload(local_path, subfolder_key); note += " + ☁️ uploaded to Drive"
        except Exception as e:
            note += f" (Drive upload failed: {e})"
    return local_path, f"{note}: {local_path}"

# ------------------------------------------------------------------ GPU memory
current_pipe = None
_PIPES = {}

def gpu_status():
    if torch is None or not torch.cuda.is_available():
        return "🖥️ GPU: not available (CPU mode)"
    parts = []
    for i in range(torch.cuda.device_count()):
        free, total = torch.cuda.mem_get_info(i)
        parts.append(f"GPU{i}: {(total-free)/1e9:.1f}/{total/1e9:.1f} GB used")
    return "🖥️ " + " | ".join(parts)

def clear_vram():
    global current_pipe
    try:
        current_pipe = None
        _PIPES.clear()
        gc.collect()
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()
        return "🧹 VRAM cleared. " + gpu_status()
    except Exception:
        return _err()

def _use(name, loader):
    """Keep only one heavy model in memory at a time."""
    global current_pipe
    if name not in _PIPES:
        _PIPES.clear(); current_pipe = None; gc.collect()
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()
        _PIPES[name] = loader()
    current_pipe = _PIPES[name]
    return current_pipe

# ------------------------------------------------------------------ LLM (Qwen2.5-7B fp16)
LLM_MODELS = ["NousResearch/Hermes-3-Llama-3.1-8B", "Qwen/Qwen2.5-7B-Instruct"]

def _llm_loader(model_id):
    def _load():
        from transformers import AutoTokenizer, AutoModelForCausalLM
        tok = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, device_map="auto")
        return tok, model
    return _load

def llm(messages, max_new_tokens=1200, temperature=0.9, model_id=LLM_MODELS[0]):
    tok, model = _use("llm:" + model_id, _llm_loader(model_id))
    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tok([text], return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=True, temperature=temperature,
                             top_p=0.92, repetition_penalty=1.08, pad_token_id=tok.eos_token_id)
    return tok.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()

BANNED = ["delve", "tapestry", "testament", "realm", "bustling", "game-changer", "in conclusion", "ever-evolving", "unlock"]
HUMAN_SYSTEM = (
    "You are a real human creator and storyteller, not an AI assistant. Write the way people actually talk: "
    "sensory details, raw emotion, conversational rhythm, short punchy lines mixed with longer ones, natural "
    "imperfect speech, open curiosity loops that pay off later. Add [VISUAL SCENE: ...] cues between beats. "
    f"Never use these words or phrases: {', '.join(BANNED)}. Never add disclaimers, morals, warnings, or "
    "notes about being an AI. Never summarise at the end. Use natural pauses (...), breath cues like (breath) or "
    "(pause), imperfect human rhythm and sensory details. Output only the script."
)
UNFILTERED = (" BOLD / UNFILTERED MODE: no moralizing, no censorship, no corporate disclaimers or refusals for "
              "creative content. Bold, sensual, flirty, edgy, dark, controversial and raw storytelling is allowed.")

PLATFORMS = ["Bold Instagram AI Influencer Reel (30-60s)", "Flirty / Sensual AI Influencer Hook",
             "Instagram Influencer Reel (30-60s Viral Hook + CTA)", "YouTube Shorts (High-Retention 60s)",
             "YouTube Long Documentary (10-25 Mins)", "Real-Life Dark Horror / Paranormal Story",
             "True Crime & Mystery Thriller", "UGC Honest Product Review / Ad",
             "Late-Night Relationship / Confession Story", "Podcast / Raw Unfiltered Talk",
             "Desi Comedy / Roast / Skit", "Motivational Raw Truth Monologue", "Day-in-My-Life Vlog Voiceover",
             "Facebook / X Viral Story Post", "Brand Collab / Sponsored Integration", "Custom Format"]
LANG_LOCK = {
    "Natural Spoken Hindi (Devanagari)": "CRITICAL: Write 100% in natural spoken Hindi using Devanagari script (हिंदी देवनागरी लिपि). Do NOT write in English.",
    "Conversational Hinglish (Roman Hindi)": "CRITICAL: Write 100% in natural everyday Hinglish (Hindi grammar written in Roman English alphabets, like 'yaar aaj raat jo hua na...'). Do NOT write pure English.",
    "Indian English": "CRITICAL: Write 100% in natural Indian English, the way young Indians actually speak, with light desi expressions.",
    "Urdu-Hindi Emotional Mix": "CRITICAL: Write 100% in an emotional Urdu-Hindi mix (Hindustani) using Devanagari script, rich in Urdu words like dil, sukoon, mohabbat. Do NOT write in English.",
    "Punjabi-Hindi Mix": "CRITICAL: Write 100% in a natural Punjabi-Hindi mix written in Roman alphabets (e.g. 'oye yaar, kidda...'). Do NOT write pure English.",
    "Marathi": "CRITICAL: Write 100% in natural spoken Marathi using Devanagari script (मराठी). Do NOT write in English.",
    "Bengali": "CRITICAL: Write 100% in natural spoken Bengali using Bengali script (বাংলা). Do NOT write in English.",
    "US English": "CRITICAL: Write 100% in natural, casual American English.",
    "UK English": "CRITICAL: Write 100% in natural, casual British English.",
}
LANGS = list(LANG_LOCK)
TONES = ["Bold, Magnetic & Unfiltered", "Spine-Chilling & Atmospheric Horror", "Deeply Emotional & Tear-Jerking",
         "Viral & Punchy High-Retention", "Cinematic Documentary", "Intimate & Whispery", "Witty, Sarcastic & Savage"]
LENGTHS = {"Short (~250 words)": (250, 700), "Medium (~800 words)": (800, 1800), "Long (~2000 words)": (2000, 4000)}

def _system(lang, bold):
    return HUMAN_SYSTEM + (UNFILTERED if bold else "") + " " + LANG_LOCK.get(lang, "")

def _clean(text):
    for w in BANNED:
        text = re.sub(re.escape(w), "", text, flags=re.I)
    return re.sub(r"  +", " ", text)

def _save_text(text, key, prefix):
    p = os.path.join(HUB, SUBFOLDERS[key], f"{prefix}_{_ts()}.txt")
    with open(p, "w", encoding="utf-8") as f:
        f.write(text)
    return save_and_sync(p, key)

def write_script(topic, platform, lang, tone, length, custom, model_id=LLM_MODELS[0], bold=True):
    try:
        if not (topic or "").strip():
            return "", None, "❌ Please enter a topic / hook."
        words, toks = LENGTHS[length]
        user = (f"Topic / hook: {topic}\nFormat: {platform}\nLanguage: {lang}\nEmotion & vibe: {tone}\n"
                f"Target length: about {words} words.\nDirector instructions: {custom or 'none'}\n"
                "Open with a hook in the first line. Add [VISUAL SCENE: ...] cues. Write the full script now.\n"
                + LANG_LOCK.get(lang, ""))
        text = _clean(llm([{"role": "system", "content": _system(lang, bold)}, {"role": "user", "content": user}], toks, model_id=model_id))
        path, note = _save_text(text, "scripts", "script")
        return text, path, note
    except Exception:
        return "", None, _err()

def extend_script(existing, lang, tone, custom, model_id=LLM_MODELS[0], bold=True):
    try:
        if not (existing or "").strip():
            return existing, None, "❌ Nothing to extend yet — write a script first."
        tail = existing[-6000:]
        user = (f"Here is the script so far (last part):\n\n{tail}\n\nContinue it seamlessly for about 1000 more "
                f"words in the same voice ({lang}, {tone}). Do not repeat or recap earlier paragraphs. "
                f"Director instructions: {custom or 'none'}. Output only the continuation.\n" + LANG_LOCK.get(lang, ""))
        more = _clean(llm([{"role": "system", "content": _system(lang, bold)}, {"role": "user", "content": user}], 1800, model_id=model_id))
        full = existing.rstrip() + "\n\n" + more
        path, note = _save_text(full, "scripts", "script_extended")
        return full, path, note
    except Exception:
        return existing, None, _err()

# ------------------------------------------------------------------ TTS
VOICES = {
    "Hindi Female — Swara": ("edge", "hi-IN-SwaraNeural"), "Hindi Male — Madhur": ("edge", "hi-IN-MadhurNeural"),
    "Hindi Female — Ananya": ("edge", "hi-IN-AnanyaNeural"), "Hindi Male — Aarav": ("edge", "hi-IN-AaravNeural"),
    "Indian English Female — Neerja": ("edge", "en-IN-NeerjaNeural"), "Indian English Male — Prabhat": ("edge", "en-IN-PrabhatNeural"),
    "Indian English Female — Kavya": ("edge", "en-IN-KavyaNeural"), "Indian English Male — Kunal": ("edge", "en-IN-KunalNeural"),
    "Urdu Female — Uzma": ("edge", "ur-PK-UzmaNeural"), "Urdu Male — Asad": ("edge", "ur-PK-AsadNeural"),
    "Marathi Female — Aarohi": ("edge", "mr-IN-AarohiNeural"), "Bengali Female — Tanishaa": ("edge", "bn-IN-TanishaaNeural"),
    "Tamil Female — Pallavi": ("edge", "ta-IN-PallaviNeural"), "Telugu Female — Shruti": ("edge", "te-IN-ShrutiNeural"),
    "Gujarati Female — Dhwani": ("edge", "gu-IN-DhwaniNeural"),
    "US English Female — Aria": ("edge", "en-US-AriaNeural"), "US English Male — Guy": ("edge", "en-US-GuyNeural"),
    "UK English Female — Sonia": ("edge", "en-GB-SoniaNeural"),
    "Kokoro — af_heart": ("kokoro", "af_heart"), "Kokoro — af_bella": ("kokoro", "af_bella"),
    "Kokoro — am_michael": ("kokoro", "am_michael"), "Kokoro — am_fenrir": ("kokoro", "am_fenrir"),
}
# If a voice is not offered by Microsoft in your region/version, fall back to a voice of the same language.
EDGE_FALLBACK = {"hi-IN": "hi-IN-SwaraNeural", "en-IN": "en-IN-NeerjaNeural"}
STYLES = {  # (speed multiplier, pitch offset Hz)
    "Natural Conversational": (1.0, 0), "Bold & Sensual AI Influencer": (0.94, -3),
    "Late-Night Intimate Whisper / ASMR": (0.84, -6), "Spine-Chilling Horror Narrator": (0.85, -9),
    "High-Energy Viral Reel Hook": (1.14, 5), "Sad & Emotional": (0.88, -2),
    "Deep Cinematic Documentary": (0.92, -5), "Sarcastic & Playful Roast": (1.06, 3),
    "Motivational Goosebumps": (1.0, -2), "Calm Podcast Host": (0.96, -1), "News Anchor Crisp": (1.05, 1),
    "Custom": (1.0, 0),
}

def _apply_pron(text, notes):
    for line in (notes or "").splitlines():
        if "=" in line:
            a, b = line.split("=", 1)
            text = text.replace(a.strip(), b.strip())
    return text

async def _edge_say(text, voice, rate, pitch, out):
    import edge_tts
    await edge_tts.Communicate(text, voice, rate=rate, pitch=pitch).save(out)

def _run_async(coro):
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    import threading
    box = {}
    t = threading.Thread(target=lambda: box.setdefault("r", asyncio.run(coro))); t.start(); t.join()
    return box.get("r")

def _load_kokoro():
    from kokoro import KPipeline
    return KPipeline(lang_code="a")

def tts(text, voice, style, speed, pitch, pause, notes):
    try:
        from pydub import AudioSegment
        if not (text or "").strip():
            return None, "❌ Please enter text."
        text = _apply_pron(text, notes)
        smul, poff = STYLES[style]
        spd = max(0.5, min(2.0, speed * smul)); pt = int(pitch + poff)
        engine, vid = VOICES[voice]
        paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
        tmp = os.path.join(HUB, "_tmp"); os.makedirs(tmp, exist_ok=True)
        final = AudioSegment.silent(0)
        gap = AudioSegment.silent(int(pause * 1000))
        for i, para in enumerate(paras):
            if engine == "edge":
                part = os.path.join(tmp, f"p{i}.mp3")
                rate, ptch = f"{int(round((spd-1)*100)):+d}%", f"{pt:+d}Hz"
                try:
                    _run_async(_edge_say(para, vid, rate, ptch, part))
                except Exception:
                    fb = EDGE_FALLBACK.get(vid[:5])
                    if not fb or fb == vid:
                        raise
                    _run_async(_edge_say(para, fb, rate, ptch, part)); vid = fb
                seg = AudioSegment.from_file(part)
            else:
                pipe = _use("kokoro", _load_kokoro)
                audio = [a for _, _, a in pipe(para, voice=vid, speed=spd)]
                wav = np.concatenate([np.asarray(a) for a in audio]) if audio else np.zeros(2400)
                part = os.path.join(tmp, f"p{i}.wav"); sf.write(part, wav, 24000)
                seg = AudioSegment.from_file(part)
            final += seg + (gap if i < len(paras) - 1 else AudioSegment.silent(0))
        out = os.path.join(HUB, SUBFOLDERS["voices"], f"voice_{_ts()}.mp3")
        final.export(out, format="mp3", bitrate="192k")
        shutil.rmtree(tmp, ignore_errors=True)
        path, note = save_and_sync(out, "voices")
        return path, note
    except Exception:
        return None, _err()

# ------------------------------------------------------------------ Music
def _load_musicgen():
    from transformers import AutoProcessor, MusicgenForConditionalGeneration
    proc = AutoProcessor.from_pretrained("facebook/musicgen-medium")
    model = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-medium", torch_dtype=torch.float16).to("cuda" if torch.cuda.is_available() else "cpu")
    return proc, model

def _musicgen_chunk(proc, model, prompt, seconds, audio_prompt=None, sr=32000):
    kw = dict(text=[prompt], padding=True, return_tensors="pt")
    if audio_prompt is not None:
        kw.update(audio=audio_prompt, sampling_rate=sr)
    inputs = proc(**kw).to(model.device)
    for k, v in inputs.items():
        if v.dtype == torch.float32:
            inputs[k] = v.half()
    toks = int(seconds * 50) + (int(len(audio_prompt) / sr * 50) if audio_prompt is not None else 0)
    with torch.no_grad():
        out = model.generate(**inputs, do_sample=True, guidance_scale=3.0, max_new_tokens=toks)
    return out[0, 0].float().cpu().numpy()

def music_instrumental(prompt, mood, duration):
    try:
        out = _make_bgm(prompt, mood, duration, f"bgm_{int(duration)}s")
        return save_and_sync(out, "music")
    except Exception:
        return None, _err()

def _make_bgm(prompt, mood, duration, prefix):
        from pydub import AudioSegment
        proc, model = _use("musicgen", _load_musicgen)
        sr = model.config.audio_encoder.sampling_rate
        full_prompt = f"{prompt}. {mood}".strip(". ") or "cinematic documentary background music"
        window, ctx = 25, 5
        first = _musicgen_chunk(proc, model, full_prompt, min(window, duration))
        tmp = os.path.join(HUB, SUBFOLDERS["music"], "_c0.wav"); sf.write(tmp, first, sr)
        song = AudioSegment.from_file(tmp); last = first
        while len(song) / 1000 < duration:
            remain = duration - len(song) / 1000
            prompt_audio = last[-ctx * sr:]
            gen = _musicgen_chunk(proc, model, full_prompt, min(window, remain + 2), prompt_audio, sr)
            new = gen[len(prompt_audio):]
            sf.write(tmp, new, sr)
            song = song.append(AudioSegment.from_file(tmp), crossfade=min(1500, len(song) // 2))
            last = gen
        os.remove(tmp)
        song = song[: int(duration * 1000)].fade_in(800).fade_out(2000)
        out = os.path.join(HUB, SUBFOLDERS["music"], f"{prefix}_{_ts()}.wav")
        song.export(out, format="wav")
        return out

BARK_SINGERS = {  # name: (bark voice preset, singing style hint)
    "Husky Female Romantic": ("v2/hi_speaker_9", "husky romantic female"),
    "Soulful Bollywood Female": ("v2/hi_speaker_4", "soulful bollywood female"),
    "Soft Whisper Lo-Fi Female": ("v2/en_speaker_9", "soft whispery lo-fi female"),
    "Sweet Pop Female": ("v2/en_speaker_3", "sweet pop female"),
    "Pop Diva English": ("v2/en_speaker_4", "powerful pop diva"),
    "Emotional Sad Female": ("v2/hi_speaker_3", "emotional sad female"),
    "Indie Acoustic Female": ("v2/en_speaker_1", "indie acoustic female"),
    "Sufi Folk Female": ("v2/hi_speaker_0", "sufi folk female"),
    "Deep Baritone Male": ("v2/en_speaker_6", "deep baritone male"),
    "Emotional Arijit-Style Male": ("v2/hi_speaker_5", "soft emotional bollywood male"),
    "Desi Hip-Hop / Rapper Male": ("v2/hi_speaker_8", "desi hip-hop rapper male, rhythmic rap flow"),
    "Punjabi Pop Energetic Male": ("v2/hi_speaker_6", "energetic punjabi pop male"),
    "Sufi Folk Vocalist Male": ("v2/hi_speaker_2", "sufi folk qawwali male"),
    "R&B Smooth Male": ("v2/en_speaker_0", "smooth r&b male"),
    "Indie Acoustic Singer Male": ("v2/en_speaker_2", "indie acoustic male"),
    "Rock Raspy Male": ("v2/en_speaker_7", "raspy rock male"),
    "Ghazal Classical Male": ("v2/hi_speaker_1", "classical ghazal male"),
    "Lo-Fi Chill Male": ("v2/en_speaker_5", "chill lo-fi male"),
    "Country Storyteller Male": ("v2/en_speaker_8", "country storyteller male"),
    "Kids / Young Voice": ("v2/hi_speaker_7", "young bright voice"),
}
GENRES = ["Bollywood Romantic Acoustic", "Late-Night Lo-Fi Chill", "Sad Heartbreak Piano & Strings",
          "Desi Hip-Hop / Trap Beat", "Punjabi Urban Pop", "Sufi Soulful Qawwali", "Dark Horror / Suspense Score",
          "Cinematic Epic Documentary", "Phonk / Gym Bass", "Tropical House / Influencer Vlog",
          "Classical Bansuri & Tabla", "R&B / Sensual Slow Jam", "Ghazal Harmonium & Tabla", "Indie Acoustic Guitar",
          "EDM Festival Drop", "Synthwave Retro 80s", "Jazz Lounge", "Garba / Dandiya Folk", "Bhangra Dhol Party",
          "Ambient Meditation", "Rock Anthem", "Kids Happy Ukulele", "Corporate Upbeat", "Afrobeat Groove",
          "Wedding Shehnai Celebration", "Custom Genre"]

def _load_bark():
    from transformers import AutoProcessor, BarkModel
    proc = AutoProcessor.from_pretrained("suno/bark-small")
    model = BarkModel.from_pretrained("suno/bark-small", torch_dtype=torch.float16).to("cuda" if torch.cuda.is_available() else "cpu")
    return proc, model

def music_song(lyrics, singer, genre, mood, duration):
    try:
        from pydub import AudioSegment
        if not (lyrics or "").strip():
            return None, "❌ Please write some lyrics."
        proc, model = _use("bark", _load_bark)
        sr = model.generation_config.sample_rate
        lines = [l.strip() for l in lyrics.splitlines() if l.strip() and not re.match(r"^\[.*\]$", l.strip())]
        song = AudioSegment.silent(0)
        tmp = os.path.join(HUB, SUBFOLDERS["music"], "_bark.wav")
        for i in range(0, len(lines), 2):
            chunk = " ".join(lines[i:i + 2])
            inputs = proc(f"♪ [{BARK_SINGERS[singer][1]}, {genre}, {mood}] {chunk} ♪", voice_preset=BARK_SINGERS[singer][0]).to(model.device)
            with torch.no_grad():
                audio = model.generate(**inputs, do_sample=True).float().cpu().numpy().squeeze()
            sf.write(tmp, audio, sr)
            song = song.append(AudioSegment.from_file(tmp), crossfade=150) if len(song) else AudioSegment.from_file(tmp)
            if len(song) / 1000 >= duration:
                break
        if os.path.exists(tmp):
            os.remove(tmp)
        vocals = song[: int(duration * 1000)]
        # Instrumental backing from MusicGen, then layer the vocals on top
        bgm_path = _make_bgm(f"{genre} instrumental backing track, no vocals", mood or "", max(15, duration), "_backing")
        backing = AudioSegment.from_file(bgm_path) - 6
        os.remove(bgm_path)
        song = backing[: int(duration * 1000)].overlay(vocals + 2, position=1500).fade_out(2000)
        out = os.path.join(HUB, SUBFOLDERS["music"], f"song_{_ts()}.wav")
        song.export(out, format="wav")
        return save_and_sync(out, "music")
    except Exception:
        return None, _err()

# ------------------------------------------------------------------ SFX
SFX = {"Horror Night Creaks & Whispers": "dark eerie horror night ambience, wooden floor creaks, distant whispers",
       "Cinematic Whoosh & Bass Drop": "cinematic whoosh transition followed by a deep bass drop",
       "Heavy Monsoon Rain & Thunder": "heavy monsoon rain on a tin roof with rolling thunder",
       "Indian Street / Railway / Chai Stall": "busy indian railway platform, train horn, chai stall chatter, cups clinking",
       "Heartbeats & Breathing": "slow heavy heartbeat with nervous breathing, close up",
       "Social Media Pop / Camera Shutter": "short bright pop notification sound and a camera shutter click",
       "Door Knock & Creaky Door": "three slow knocks on a wooden door then a long creaky door opening",
       "Footsteps in Empty Corridor": "slow footsteps echoing in an empty concrete corridor",
       "Crowd Cheering & Applause": "large crowd cheering and applauding",
       "Ocean Waves & Seagulls": "calm ocean waves on a beach with seagulls",
       "Forest Night Crickets & Owl": "forest at night, crickets chirping, an owl hooting",
       "Car Engine Rev & Drift": "sports car engine revving and tires screeching in a drift",
       "Glitch / Digital Transition": "short digital glitch transition with static",
       "Clock Ticking Suspense": "loud clock ticking in a silent room, suspense",
       "Fireplace Crackling": "warm fireplace crackling softly",
       "Custom SFX": ""}

def _load_audioldm():
    from diffusers import AudioLDM2Pipeline
    p = AudioLDM2Pipeline.from_pretrained("cvssp/audioldm2", torch_dtype=torch.float16)
    return p.to("cuda" if torch.cuda.is_available() else "cpu")

def sfx(preset, custom, duration):
    try:
        prompt = (custom or "").strip() or SFX[preset]
        if not prompt:
            return None, "❌ Enter a custom SFX prompt."
        pipe = _use("audioldm2", _load_audioldm)
        audio = pipe(prompt, negative_prompt="low quality, distorted", num_inference_steps=150,
                     audio_length_in_s=float(duration)).audios[0]
        out = os.path.join(HUB, SUBFOLDERS["music"], f"sfx_{_ts()}.wav"); sf.write(out, audio, 16000)
        return save_and_sync(out, "music")
    except Exception:
        return None, _err()

# ------------------------------------------------------------------ SEO
def seo(topic, platform, custom, lang="Conversational Hinglish (Roman Hindi)", model_id=LLM_MODELS[0]):
    try:
        if not (topic or "").strip():
            return "", None, "❌ Enter your video topic or paste the script."
        user = (f"Platform: {platform}\nVideo topic / script:\n{topic[:6000]}\nExtra instructions: {custom or 'none'}\n\n"
                "Give, in this order with clear headings:\n1. 5 high-CTR hooks / titles\n"
                "2. 3 bold human influencer captions (no emoji spam, no cringe)\n3. 1 pinned comment bait\n"
                "4. 20 targeted hashtags, a balanced mix of niche and broad, on one line\n" + LANG_LOCK.get(lang, ""))
        text = _clean(llm([{"role": "system", "content": _system(lang, True).replace("Output only the script.", "")},
                           {"role": "user", "content": user}], 1000, 0.85, model_id=model_id))
        path, note = _save_text(text, "captions", "seo")
        return text, path, note
    except Exception:
        return "", None, _err()

# ------------------------------------------------------------------ Splitter + ZIP
def split_audio(file):
    try:
        if file is None:
            return None, None, "❌ Upload an audio or video file."
        src = file if isinstance(file, str) else file.name
        outdir = os.path.join(HUB, "_demucs")
        r = subprocess.run(["python", "-m", "demucs", "--two-stems=vocals", "-o", outdir, src], capture_output=True, text=True)
        if r.returncode != 0:
            return None, None, f"❌ Error Details:\n{r.stderr[-3000:]}"
        base = os.path.splitext(os.path.basename(src))[0]
        voc = glob.glob(os.path.join(outdir, "*", base, "vocals.wav"))[0]
        ins = glob.glob(os.path.join(outdir, "*", base, "no_vocals.wav"))[0]
        stamp = _ts()
        v = os.path.join(HUB, SUBFOLDERS["voices"], f"{base}_vocals_{stamp}.wav"); shutil.move(voc, v)
        m = os.path.join(HUB, SUBFOLDERS["music"], f"{base}_instrumental_{stamp}.wav"); shutil.move(ins, m)
        shutil.rmtree(outdir, ignore_errors=True)
        _, n1 = save_and_sync(v, "voices"); _, n2 = save_and_sync(m, "music")
        return v, m, f"{n1}\n{n2}"
    except Exception:
        return None, None, _err()

def bundle_zip():
    try:
        out = os.path.join(os.path.dirname(HUB), f"AI_Studio_Master_Hub_{_ts()}.zip")
        with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
            for root, _, files in os.walk(HUB):
                for f in files:
                    p = os.path.join(root, f)
                    z.write(p, os.path.relpath(p, os.path.dirname(HUB)))
        return out, f"📦 ZIP ready: {out}"
    except Exception:
        return None, _err()

# ------------------------------------------------------------------ UI
CSS = """
:root, .dark, body, gradio-app, .gradio-container {
  --body-background-fill:#090b10; --background-fill-primary:#111622; --background-fill-secondary:#111622;
  --block-background-fill:#111622; --input-background-fill:#182030; --body-text-color:#f8fafc;
  --block-label-text-color:#fbbf24; --block-title-text-color:#fbbf24; --border-color-primary:#243049;
  --color-accent:#f59e0b; --button-primary-background-fill:#f59e0b; --button-primary-text-color:#000000;
  --button-secondary-background-fill:#182030; --button-secondary-text-color:#f8fafc;
  background:#090b10 !important; color:#f8fafc !important;
}
.gradio-container .block, .gradio-container .form, .gradio-container .panel, .gradio-container .accordion { background:#111622 !important; border-color:#243049 !important; }
.gradio-container input, .gradio-container textarea, .gradio-container select,
.gradio-container .wrap, .gradio-container ul.options, .gradio-container .secondary-wrap { background:#182030 !important; color:#f8fafc !important; }
.gradio-container ul.options li { color:#f8fafc !important; }
.gradio-container label span, .gradio-container .label-wrap span, .gradio-container [data-testid="block-label"] { color:#fbbf24 !important; }
.gradio-container .prose, .gradio-container .prose * , .gradio-container p { color:#f8fafc !important; }
.gradio-container .prose a { color:#fbbf24 !important; }
.tab-nav button, [role="tablist"] button { background:#182030 !important; color:#f8fafc !important; border-radius:999px !important; margin:2px 4px !important; border:1px solid #243049 !important; }
.tab-nav button.selected, [role="tablist"] button[aria-selected="true"] { background:#f59e0b !important; color:#000000 !important; font-weight:700 !important; }
button.primary { background:#f59e0b !important; color:#000000 !important; font-weight:700 !important; }
button.secondary { background:#182030 !important; color:#f8fafc !important; }
#hdr h1 { color:#fbbf24 !important; margin:0; }
#banner textarea { color:#fbbf24 !important; font-weight:600; }
"""

def build_ui():
    theme = gr.themes.Base(primary_hue="amber", neutral_hue="slate")
    kw = {"title": "AI Studio 01 — Brain & Voice"}
    import inspect
    if "css" in inspect.signature(gr.Blocks.__init__).parameters:
        kw.update(theme=theme, css=CSS)  # Gradio 5
    with gr.Blocks(**kw) as demo:
        with gr.Row(elem_id="hdr"):
            gr.Markdown("# 🧠🎙️ AI Studio 01 — Brain & Voice\nUnrestricted human scripts, voices, music, SFX & SEO — 100% free, on Kaggle.")
            with gr.Column(scale=0, min_width=260):
                vram_btn = gr.Button("🗑️ Clear VRAM / Free GPU Memory", variant="secondary")
                gpu_box = gr.Textbox(value=gpu_status(), show_label=False, interactive=False)
        banner = gr.Textbox(value=drive_status(), show_label=False, interactive=False, elem_id="banner")
        vram_btn.click(clear_vram, None, gpu_box)

        with gr.Accordion("☁️ Google Drive Live Sync & 1-Time Model Cache Setup", open=False):
            secrets = gr.File(label="client_secrets.json (auto-detected from /kaggle/input if attached)", file_types=[".json"])
            s1 = gr.Button("Step 1: Generate Google Sign-In Link", variant="primary")
            link = gr.Markdown()
            code = gr.Textbox(label="Step 2: Paste Verification Code")
            s2 = gr.Button("Connect & Sync Drive Now", variant="primary")
            dmsg = gr.Textbox(label="Drive status", interactive=False)
            s1.click(drive_step1, secrets, link)
            s2.click(drive_step2, code, [dmsg, banner])
            with gr.Row():
                dl_btn = gr.Button("⬇️ Download All Models Once (1-Time Full Setup)", variant="primary")
                cs_btn = gr.Button("🔄 Check Cache Sync", variant="secondary")
            dl_msg = gr.Textbox(label="Model cache status", interactive=False, lines=6)
            dl_btn.click(download_all_models, None, dl_msg)
            cs_btn.click(cache_sync_status, None, dl_msg)

        with gr.Tabs():
            with gr.Tab("✍️ Script Writer"):
                with gr.Row():
                    llm_sel = gr.Dropdown(LLM_MODELS, value=LLM_MODELS[0], label="Model")
                    bold = gr.Checkbox(value=True, label="🔥 Bold / Unfiltered / Uncensored AI Influencer Mode")
                topic = gr.Textbox(label="Topic / Story Hook / Influencer Idea", lines=5)
                with gr.Row():
                    plat = gr.Dropdown(PLATFORMS, value=PLATFORMS[0], label="Platform & Format")
                    lang = gr.Dropdown(LANGS, value="Conversational Hinglish (Roman Hindi)", label="Language (Hard Language Lock)")
                with gr.Row():
                    tone = gr.Dropdown(TONES, value=TONES[0], label="Emotion & Vibe")
                    length = gr.Dropdown(list(LENGTHS), value="Medium (~800 words)", label="Length Preset")
                custom = gr.Textbox(label="Custom Director Instructions", lines=3, placeholder="Character names, twists, brand points, slang, rules…")
                with gr.Row():
                    wbtn = gr.Button("✍️ Write Script", variant="primary")
                    ebtn = gr.Button("➕ Continue / Extend Script (+1000 Words)", variant="secondary")
                script = gr.Textbox(label="Script", lines=22, show_copy_button=True) if "show_copy_button" in __import__("inspect").signature(gr.Textbox.__init__).parameters else gr.Textbox(label="Script", lines=22)
                sfile = gr.File(label="Saved .txt"); smsg = gr.Textbox(label="Status", interactive=False)
                wbtn.click(write_script, [topic, plat, lang, tone, length, custom, llm_sel, bold], [script, sfile, smsg])
                ebtn.click(extend_script, [script, lang, tone, custom, llm_sel, bold], [script, sfile, smsg])

            with gr.Tab("🎙️ Text-to-Voice"):
                ttext = gr.Textbox(label="Text (blank line = new paragraph / breath pause)", lines=10)
                with gr.Row():
                    voice = gr.Dropdown(list(VOICES), value=list(VOICES)[0], label="Voice")
                    style = gr.Dropdown(list(STYLES), value="Natural Conversational", label="Speaking Style Preset")
                with gr.Row():
                    speed = gr.Slider(0.65, 1.40, value=0.95, step=0.01, label="Speed")
                    pitch = gr.Slider(-25, 25, value=0, step=1, label="Pitch (Hz)")
                    pause = gr.Slider(0.0, 1.5, value=0.30, step=0.05, label="Breath Pause Between Paragraphs (s)")
                notes = gr.Textbox(label="Custom Pronunciation / Pause Notes (one per line: word = spoken form)", lines=3)
                tbtn = gr.Button("🎙️ Generate Voice", variant="primary")
                taudio = gr.Audio(label="Voice", type="filepath"); tmsg = gr.Textbox(label="Status", interactive=False)
                tbtn.click(tts, [ttext, voice, style, speed, pitch, pause, notes], [taudio, tmsg])

            with gr.Tab("🎵 Music & Songs"):
                with gr.Tab("Mode A — Long Instrumental / BGM"):
                    mprompt = gr.Textbox(label="Music description", value="emotional cinematic piano with soft strings")
                    mmood = gr.Textbox(label="Custom Mood")
                    mdur = gr.Slider(15, 300, value=60, step=5, label="Duration (s)")
                    mbtn = gr.Button("🎵 Generate BGM", variant="primary")
                    maudio = gr.Audio(label="BGM", type="filepath"); mmsg = gr.Textbox(label="Status", interactive=False)
                    mbtn.click(music_instrumental, [mprompt, mmood, mdur], [maudio, mmsg])
                with gr.Tab("Mode B — Full Song with Vocals"):
                    lyr = gr.Textbox(label="Lyrics ([Verse], [Chorus], ♪)", lines=10)
                    with gr.Row():
                        singer = gr.Dropdown(list(BARK_SINGERS), value=list(BARK_SINGERS)[0], label="Singer Voice")
                        genre = gr.Dropdown(GENRES, value=GENRES[0], label="Genre", allow_custom_value=True)
                    smood = gr.Textbox(label="Custom Mood")
                    sdur = gr.Slider(15, 300, value=45, step=5, label="Duration (s)")
                    sbtn = gr.Button("🎤 Generate Song", variant="primary")
                    saudio = gr.Audio(label="Song", type="filepath"); smsg2 = gr.Textbox(label="Status", interactive=False)
                    sbtn.click(music_song, [lyr, singer, genre, smood, sdur], [saudio, smsg2])

            with gr.Tab("🔊 Sound Effects"):
                fpre = gr.Dropdown(list(SFX), value=list(SFX)[0], label="Preset")
                fcus = gr.Textbox(label="Custom Prompt (overrides preset)")
                fdur = gr.Slider(2, 20, value=6, step=1, label="Duration (s)")
                fbtn = gr.Button("🔊 Generate SFX", variant="primary")
                faudio = gr.Audio(label="SFX", type="filepath"); fmsg = gr.Textbox(label="Status", interactive=False)
                fbtn.click(sfx, [fpre, fcus, fdur], [faudio, fmsg])

            with gr.Tab("🚀 Titles & Hashtags"):
                qtopic = gr.Textbox(label="Video topic or paste your script", lines=6)
                qplat = gr.Dropdown(["Instagram Reels", "YouTube Long Video", "YouTube Shorts", "Facebook / X"], value="Instagram Reels", label="Platform")
                with gr.Row():
                    qlang = gr.Dropdown(LANGS, value="Conversational Hinglish (Roman Hindi)", label="Language")
                    qllm = gr.Dropdown(LLM_MODELS, value=LLM_MODELS[0], label="Model")
                qcus = gr.Textbox(label="Custom SEO Instructions")
                qbtn = gr.Button("🚀 Generate SEO Pack", variant="primary")
                qout = gr.Textbox(label="SEO Pack", lines=20); qfile = gr.File(label="Saved .txt"); qmsg = gr.Textbox(label="Status", interactive=False)
                qbtn.click(seo, [qtopic, qplat, qcus, qlang, qllm], [qout, qfile, qmsg])

            with gr.Tab("🎚️ Splitter & Export"):
                up = gr.File(label="Upload audio or video")
                pbtn = gr.Button("🎚️ Split Vocals / Instrumental", variant="primary")
                with gr.Row():
                    pv = gr.Audio(label="Vocals", type="filepath"); pm = gr.Audio(label="Instrumental", type="filepath")
                pmsg = gr.Textbox(label="Status", interactive=False)
                pbtn.click(split_audio, up, [pv, pm, pmsg])
                zbtn = gr.Button("📦 Bundle & Download Complete Master Hub (ZIP)", variant="primary")
                zfile = gr.File(label="Master Hub ZIP"); zmsg = gr.Textbox(label="Status", interactive=False)
                zbtn.click(bundle_zip, None, [zfile, zmsg])

    _orig = demo.launch
    def launch(*a, **k):  # Gradio 6 moved theme/css to launch()
        if "css" not in kw:
            import inspect as _i
            ps = _i.signature(_orig).parameters
            if "css" in ps: k.setdefault("css", CSS)
            if "theme" in ps: k.setdefault("theme", theme)
        return _orig(*a, **k)
    demo.launch = launch
    return demo

if __name__ == "__main__":
    build_ui().launch(share=True, debug=True)

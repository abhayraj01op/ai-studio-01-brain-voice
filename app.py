"""AI Studio 01 — Brain & Voice (Ultra Edition — Kaggle T4 x2, Python 3.13, zero paid APIs)."""
import os, gc, re, json, glob, shutil, asyncio, traceback, subprocess, datetime, zipfile, threading
import warnings
warnings.filterwarnings("ignore")

# Disable Xet background writer (prevents Kaggle disk RuntimeError) & set persistent cache
os.environ["HF_HUB_DISABLE_XET"] = "1"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
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

def disk_status():
    try:
        target = "/kaggle/working" if os.path.isdir("/kaggle/working") else os.getcwd()
        total, used, free = shutil.disk_usage(target)
        return f"💾 Disk: {used/1e9:.1f}/{total/1e9:.1f} GB used ({free/1e9:.1f} GB free)"
    except Exception:
        return "💾 Disk: unknown"

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
        return (
            f"✅ Connected in 2 seconds! Hub + 9 subfolders ready on Drive. Synced {n} existing files.\n"
            f"♻️ Smart Disk-Safe Mode active: Models load on-demand from Drive without filling Kaggle's 19.5 GB disk."
        ), drive_status()
    except Exception:
        return _err(), drive_status()

def _list_children(parent):
    return DRIVE["drive"].ListFile({"q": f"'{parent}' in parents and trashed=false"}).GetList()

def _free_local_model_cache(keep_model_ids=None):
    """Delete cached HuggingFace model folders from local Kaggle disk except keep_model_ids."""
    keep_folders = {"models--" + m.replace("/", "--") for m in (keep_model_ids or [])}
    hub_cache = os.path.join(HUB, SUBFOLDERS["cache"], "hub")
    if not os.path.isdir(hub_cache):
        return
    for entry in os.listdir(hub_cache):
        if entry.startswith("models--") and entry not in keep_folders:
            shutil.rmtree(os.path.join(hub_cache, entry), ignore_errors=True)

def _ensure_disk_space_for(model_id, min_free_gb=14.0):
    """Before downloading/restoring a heavy model, free older local model caches if Kaggle disk is tight."""
    try:
        target = "/kaggle/working" if os.path.isdir("/kaggle/working") else os.getcwd()
        _, _, free = shutil.disk_usage(target)
        if free / 1e9 < min_free_gb:
            _free_local_model_cache(keep_model_ids=[model_id])
    except Exception:
        pass

def _ensure_model_from_drive(model_id):
    """Auto-restore a single model folder from Drive on demand if not on local disk yet."""
    _ensure_disk_space_for(model_id, min_free_gb=14.0)
    if not DRIVE["drive"]:
        return
    folder_name = "models--" + model_id.replace("/", "--")
    local_dir = os.path.join(HUB, SUBFOLDERS["cache"], "hub", folder_name)
    if os.path.isdir(local_dir) and any(os.scandir(local_dir)):
        return
    try:
        hub_folders = [f for f in _list_children(DRIVE["ids"]["cache"]) if f["title"] == "hub"]
        if not hub_folders:
            return
        model_folders = [f for f in _list_children(hub_folders[0]["id"]) if f["title"] == folder_name]
        if not model_folders:
            return
        def walk(fid, dest):
            os.makedirs(dest, exist_ok=True)
            for f in _list_children(fid):
                target = os.path.join(dest, f["title"])
                if f["mimeType"] == "application/vnd.google-apps.folder":
                    walk(f["id"], target)
                elif not os.path.exists(target):
                    f.GetContentFile(target)
        walk(model_folders[0]["id"], local_dir)
    except Exception:
        pass

CACHE_SYNC = {"running": False, "msg": "Model cache not synced yet."}

def _sync_cache_to_drive_blocking():
    """Mirror 99_model_cache to Drive synchronously (existing same-size files skipped)."""
    if not DRIVE["drive"]:
        return 0
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
    return n

ALL_MODELS = [
    "NousResearch/Hermes-3-Llama-3.1-8B",
    "mistralai/Mistral-Nemo-Instruct-2407",
    "Qwen/Qwen2.5-14B-Instruct",
    "Qwen/Qwen2.5-7B-Instruct",
    "facebook/musicgen-medium",
    "facebook/musicgen-large",
    "suno/bark",
    "suno/bark-small",
    "cvssp/audioldm2",
]

DEFAULT_CACHE_MODELS = [
    "NousResearch/Hermes-3-Llama-3.1-8B",
    "mistralai/Mistral-Nemo-Instruct-2407",
    "facebook/musicgen-medium",
    "suno/bark",
    "cvssp/audioldm2",
]

def download_selected_models(selected_models):
    """Disk-safe sequential downloader: downloads 1 model -> uploads to Drive -> frees Kaggle 19.5GB disk -> next model."""
    try:
        from huggingface_hub import snapshot_download
        models = selected_models or DEFAULT_CACHE_MODELS
        lines = []
        hub_cache = os.path.join(HUB, SUBFOLDERS["cache"], "hub")
        for idx, m in enumerate(models):
            _free_local_model_cache(keep_model_ids=[])
            CACHE_SYNC["msg"] = f"⏳ [{idx+1}/{len(models)}] Downloading {m}..."
            snapshot_download(
                m,
                allow_patterns=["*.json", "*.safetensors", "*.bin", "*.model", "*.txt", "*.py", "*.npz", "*.tiktoken", "tokenizer*"],
                max_workers=2,
            )
            if DRIVE["drive"]:
                CACHE_SYNC["msg"] = f"☁️ [{idx+1}/{len(models)}] Uploading {m} to Google Drive..."
                _sync_cache_to_drive_blocking()
                folder_name = "models--" + m.replace("/", "--")
                shutil.rmtree(os.path.join(hub_cache, folder_name), ignore_errors=True)
                lines.append(f"✅ {m} -> Saved to Google Drive & local disk freed!")
            else:
                lines.append(f"✅ {m} -> Downloaded locally (connect Drive to backup).")
        if DRIVE["drive"] and models:
            CACHE_SYNC["msg"] = f"♻️ Restoring primary model ({models[0]}) to GPU disk..."
            _ensure_model_from_drive(models[0])
            lines.append(f"🚀 {models[0]} ready on GPU disk for instant use!")
        CACHE_SYNC["msg"] = "✅ Selected models cached safely to Google Drive!"
        return "\n".join(lines)
    except Exception:
        return _err()

def cache_sync_status():
    return f"{CACHE_SYNC['msg']} | {disk_status()}"

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

# ------------------------------------------------------------------ GPU & Disk memory manager
current_pipe = None
_PIPES = {}

def gpu_status():
    if torch is None or not torch.cuda.is_available():
        return "🖥️ GPU: CPU Mode | " + disk_status()
    parts = []
    for i in range(torch.cuda.device_count()):
        free, total = torch.cuda.mem_get_info(i)
        parts.append(f"GPU{i}: {(total-free)/1e9:.1f}/{total/1e9:.1f}GB")
    return "🖥️ " + " | ".join(parts) + " | " + disk_status()

def clear_vram_and_disk():
    global current_pipe
    try:
        current_pipe = None
        _PIPES.clear()
        gc.collect()
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()
        _free_local_model_cache(keep_model_ids=[])
        return "🧹 VRAM + Temp Model Disk cleared! " + gpu_status()
    except Exception:
        return _err()

def _use(name, loader):
    """Keep only one heavy model in GPU VRAM at a time to prevent OOM when switching tools."""
    global current_pipe
    if name not in _PIPES:
        _PIPES.clear(); current_pipe = None; gc.collect()
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()
        _PIPES[name] = loader()
    current_pipe = _PIPES[name]
    return current_pipe

# ------------------------------------------------------------------ LLMs (Upgraded Heavy Models)
LLM_MODELS = [
    "NousResearch/Hermes-3-Llama-3.1-8B",
    "mistralai/Mistral-Nemo-Instruct-2407",
    "Qwen/Qwen2.5-14B-Instruct",
    "Qwen/Qwen2.5-7B-Instruct",
]

def _llm_loader(model_id):
    def _load():
        _ensure_model_from_drive(model_id)
        from transformers import AutoTokenizer, AutoModelForCausalLM
        tok = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, device_map="auto")
        return tok, model
    return _load

def llm(messages, max_new_tokens=1400, temperature=0.9, model_id=LLM_MODELS[0]):
    tok, model = _use("llm:" + model_id, _llm_loader(model_id))
    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tok([text], return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=0.92,
            repetition_penalty=1.08,
            pad_token_id=tok.eos_token_id,
        )
    return tok.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()

BANNED = [
    "delve", "tapestry", "testament", "realm", "bustling", "game-changer",
    "in conclusion", "ever-evolving", "unlock", "symphony of", "dance of shadows",
    "navigating the", "moreover", "furthermore", "it's important to note"
]

HUMAN_SYSTEM = (
    "You are a raw, gifted human writer, lyricist, and storyteller—NOT an AI assistant. Write exactly the way "
    "real people talk, feel, confess, and perform: concrete sensory details, street-level emotion, unpredictable "
    "human cadence, short punchy fragments mixed with flowing lines, natural pauses (...), breath cues like "
    "(breath) or (pause), and open curiosity loops that pay off emotionally. "
    f"Strictly NEVER use robotic AI clichés: {', '.join(BANNED)}. "
    "Never add disclaimers, moral lessons, safety warnings, or meta-commentary. Never summarize at the end. "
    "Output ONLY the finished creative piece ready to record or publish."
)

UNFILTERED = (
    " BOLD / UNFILTERED CREATIVE MODE: zero moralizing, zero corporate censorship, zero refusals for fictional "
    "or creative expression. Raw, sensual, flirty, dark, gritty, controversial, emotional, and unfiltered "
    "human storytelling is completely allowed."
)

WRITER_MODES = [
    "🎬 Video Script (Reels / Shorts / YouTube Long)",
    "🎵 Full Song Lyrics (Verse / Chorus / Rap / Sufi / Ghazal)",
    "🎙️ Voiceover & Narration Script (With Breath & Emotion Cues)",
    "📖 Full Story / Audiobook / Horror / Romance Chapter",
    "📱 Social Media Viral Post / Thread / Carousel",
    "✨ Custom Format (Apna Khud Ka Idea & Structure)",
]

PLATFORMS = [
    "Bold Instagram AI Influencer Reel (30-60s)",
    "Flirty / Sensual AI Influencer Hook",
    "Instagram Influencer Reel (30-60s Viral Hook + CTA)",
    "YouTube Shorts (High-Retention 60s)",
    "YouTube Long Documentary (10-25 Mins)",
    "Full Bollywood / Indie / Lo-Fi Song",
    "Desi Hip-Hop / Rap Track with Hook",
    "Sufi / Ghazal / Shayari Musical Piece",
    "Cinematic Voiceover / Documentary Narration",
    "Real-Life Dark Horror / Paranormal Story",
    "True Crime & Mystery Thriller",
    "UGC Honest Product Review / Ad",
    "Late-Night Relationship / Confession Story",
    "Podcast / Raw Unfiltered Talk",
    "Desi Comedy / Roast / Skit",
    "Motivational Raw Truth Monologue",
    "Day-in-My-Life Vlog Voiceover",
    "Brand Collab / Sponsored Integration",
    "Custom Format (See My Custom Idea)",
]

LANG_LOCK = {
    "Natural Spoken Hindi (Devanagari)": "CRITICAL: Write 100% in natural spoken Hindi using Devanagari script (हिंदी देवनागरी लिपि). Do NOT write in English.",
    "Conversational Hinglish (Roman Hindi)": "CRITICAL: Write 100% in natural everyday Hinglish (Hindi grammar written in Roman English alphabets, like 'yaar aaj raat jo hua na...'). Do NOT write pure English.",
    "Indian English": "CRITICAL: Write 100% in natural Indian English, the way young Indians actually speak, with light desi expressions.",
    "Urdu-Hindi Emotional Mix": "CRITICAL: Write 100% in an emotional Urdu-Hindi mix (Hindustani) using Devanagari script, rich in Urdu words like dil, sukoon, mohabbat, khamoshi. Do NOT write in English.",
    "Punjabi-Hindi Mix": "CRITICAL: Write 100% in a natural Punjabi-Hindi mix written in Roman alphabets (e.g. 'oye yaar, kidda...'). Do NOT write pure English.",
    "Marathi": "CRITICAL: Write 100% in natural spoken Marathi using Devanagari script (मराठी). Do NOT write in English.",
    "Bengali": "CRITICAL: Write 100% in natural spoken Bengali using Bengali script (বাংলা). Do NOT write in English.",
    "US English": "CRITICAL: Write 100% in natural, casual American English.",
    "UK English": "CRITICAL: Write 100% in natural, casual British English.",
}
LANGS = list(LANG_LOCK)

TONES = [
    "Bold, Magnetic & Unfiltered",
    "Romantic, Soulful & Heart-Touching",
    "Spine-Chilling & Atmospheric Horror",
    "Deeply Emotional & Tear-Jerking",
    "Viral & Punchy High-Retention",
    "Cinematic Documentary",
    "Intimate, Late-Night & Whispery",
    "Witty, Sarcastic & Savage",
    "Energetic, Desi Swag & Hype",
]

LENGTHS = {
    "Short (~250 words / 30-60s)": (250, 750),
    "Medium (~800 words / 3-5 mins)": (800, 1900),
    "Long (~2000 words / 10-15 mins)": (2000, 4000),
}

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

def write_script(mode, topic, my_idea, platform, lang, tone, length, custom_rules, model_id=LLM_MODELS[0], bold=True):
    try:
        if not (topic or "").strip() and not (my_idea or "").strip():
            return "", None, "❌ Please enter a Topic or write your own Idea / Storyline."
        words, toks = LENGTHS[length]

        mode_instructions = {
            "🎬 Video Script (Reels / Shorts / YouTube Long)": (
                "Write a high-retention video script. Open with an irresistible scroll-stopping hook in the first 3 seconds. "
                "Include [VISUAL SCENE: ...] cues between beats so it is ready for video generation."
            ),
            "🎵 Full Song Lyrics (Verse / Chorus / Rap / Sufi / Ghazal)": (
                "Write complete, musical, singable song lyrics with clear sections like [Intro], [Verse 1], [Pre-Chorus], "
                "[Chorus / Hook], [Verse 2], [Bridge / Rap], and [Outro]. Focus on natural rhyme, rhythm, syllable flow, "
                "and deep emotional imagery so a singer can sing it effortlessly."
            ),
            "🎙️ Voiceover & Narration Script (With Breath & Emotion Cues)": (
                "Write a pure spoken-word voiceover script ready for TTS/recording. Do NOT add [VISUAL SCENE] tags. "
                "Instead use natural spoken punctuation (...), paragraph breaks for breath pauses, and subtle delivery cues "
                "so the voice sounds 100% human and intimate."
            ),
            "📖 Full Story / Audiobook / Horror / Romance Chapter": (
                "Write an immersive, gripping narrative story with vivid scenes, realistic human dialogue, tension, "
                "and emotional payoff."
            ),
            "📱 Social Media Viral Post / Thread / Carousel": (
                "Write a punchy, high-engagement social media post/thread with a strong opening line, scannable beats, "
                "and a natural closing call-to-conversation."
            ),
            "✨ Custom Format (Apna Khud Ka Idea & Structure)": (
                "Follow the user's custom idea, structure, and format instructions strictly."
            ),
        }

        user = (
            f"Mode: {mode}\n"
            f"Format / Sub-style: {platform}\n"
            f"Main Topic / Title: {topic or 'See My Own Idea below'}\n"
            f"My Own Core Idea / Storyline / Hook / Rough Notes:\n{my_idea or 'Use your best creative direction'}\n\n"
            f"Language: {lang}\n"
            f"Emotion & Vibe: {tone}\n"
            f"Target Length: ~{words} words.\n"
            f"Mode Rules: {mode_instructions.get(mode, '')}\n"
            f"Extra Director / Lyrics / Custom Instructions: {custom_rules or 'none'}\n\n"
            "Write the complete piece now. Make it sound 100% raw, authentic, and human.\n"
            + LANG_LOCK.get(lang, "")
        )
        text = _clean(llm([{"role": "system", "content": _system(lang, bold)}, {"role": "user", "content": user}], toks, model_id=model_id))
        path, note = _save_text(text, "scripts", "script")
        return text, path, note
    except Exception:
        return "", None, _err()

def extend_script(existing, mode, lang, tone, my_idea, custom_rules, model_id=LLM_MODELS[0], bold=True):
    try:
        if not (existing or "").strip():
            return existing, None, "❌ Nothing to extend yet — write something first."
        tail = existing[-6000:]
        user = (
            f"Here is the piece so far (last part):\n\n{tail}\n\n"
            f"Continue it seamlessly for about 1000 more words in the exact same style ({mode}, {lang}, {tone}). "
            f"If it is a song, continue with the next Verse, Bridge, Rap, or Outro. Do not repeat or recap earlier lines. "
            f"My Idea context: {my_idea or 'same'}. Extra instructions: {custom_rules or 'none'}. Output only the continuation.\n"
            + LANG_LOCK.get(lang, "")
        )
        more = _clean(llm([{"role": "system", "content": _system(lang, bold)}, {"role": "user", "content": user}], 1800, model_id=model_id))
        full = existing.rstrip() + "\n\n" + more
        path, note = _save_text(full, "scripts", "script_extended")
        return full, path, note
    except Exception:
        return existing, None, _err()

def humanize_polish(existing, lang, custom_polish, model_id=LLM_MODELS[0], bold=True):
    try:
        if not (existing or "").strip():
            return existing, None, "❌ Paste or generate text first before polishing."
        user = (
            "Rewrite and polish the text below so it sounds 100% like a real human speaking or singing from the heart—zero AI stiffness, "
            "zero textbook phrasing, better spoken rhythm, sharper hooks, natural pauses (...), and raw desi/human authenticity. "
            f"Keep the exact meaning and format. Custom polish direction: {custom_polish or 'make it sound ultra-natural and human'}.\n\n"
            f"TEXT TO HUMANIZE:\n{existing[:7000]}\n\n"
            + LANG_LOCK.get(lang, "")
        )
        polished = _clean(llm([{"role": "system", "content": _system(lang, bold)}, {"role": "user", "content": user}], 2200, temperature=0.85, model_id=model_id))
        path, note = _save_text(polished, "scripts", "humanized")
        return polished, path, note
    except Exception:
        return existing, None, _err()

def send_to_voice(script_text):
    cleaned = re.sub(r"\[VISUAL SCENE:.*?\]", "", script_text or "", flags=re.I | re.S)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
    return cleaned, "✅ Script sent to Text-to-Voice tab (with [VISUAL SCENE] tags cleaned)!"

def send_to_song(script_text):
    return (script_text or "").strip(), "✅ Lyrics sent to Music & Songs (Mode B) tab!"

def send_to_seo(script_text):
    return (script_text or "").strip(), "✅ Content sent to Titles & Hashtags tab!"

# ------------------------------------------------------------------ TTS
VOICES = {
    "Hindi Female — Swara": ("edge", "hi-IN-SwaraNeural"),
    "Hindi Male — Madhur": ("edge", "hi-IN-MadhurNeural"),
    "Hindi Female — Ananya": ("edge", "hi-IN-AnanyaNeural"),
    "Hindi Male — Aarav": ("edge", "hi-IN-AaravNeural"),
    "Indian English Female — Neerja": ("edge", "en-IN-NeerjaNeural"),
    "Indian English Male — Prabhat": ("edge", "en-IN-PrabhatNeural"),
    "Indian English Female — Kavya": ("edge", "en-IN-KavyaNeural"),
    "Indian English Male — Kunal": ("edge", "en-IN-KunalNeural"),
    "Urdu Female — Uzma": ("edge", "ur-PK-UzmaNeural"),
    "Urdu Male — Asad": ("edge", "ur-PK-AsadNeural"),
    "Marathi Female — Aarohi": ("edge", "mr-IN-AarohiNeural"),
    "Bengali Female — Tanishaa": ("edge", "bn-IN-TanishaaNeural"),
    "Tamil Female — Pallavi": ("edge", "ta-IN-PallaviNeural"),
    "Telugu Female — Shruti": ("edge", "te-IN-ShrutiNeural"),
    "Gujarati Female — Dhwani": ("edge", "gu-IN-DhwaniNeural"),
    "US English Female — Aria": ("edge", "en-US-AriaNeural"),
    "US English Male — Guy": ("edge", "en-US-GuyNeural"),
    "UK English Female — Sonia": ("edge", "en-GB-SoniaNeural"),
    "Kokoro — af_heart (Warm US Female)": ("kokoro", "af_heart"),
    "Kokoro — af_bella (Expressive US Female)": ("kokoro", "af_bella"),
    "Kokoro — af_nicole (Whispery ASMR Female)": ("kokoro", "af_nicole"),
    "Kokoro — am_michael (Deep US Male)": ("kokoro", "am_michael"),
    "Kokoro — am_fenrir (Intense Storyteller Male)": ("kokoro", "am_fenrir"),
    "Kokoro — bf_emma (British Female)": ("kokoro", "bf_emma"),
}
EDGE_FALLBACK = {"hi-IN": "hi-IN-SwaraNeural", "en-IN": "en-IN-NeerjaNeural"}
STYLES = {
    "Natural Conversational": (1.0, 0),
    "Bold & Sensual AI Influencer": (0.94, -3),
    "Late-Night Intimate Whisper / ASMR": (0.84, -6),
    "Spine-Chilling Horror Narrator": (0.85, -9),
    "High-Energy Viral Reel Hook": (1.14, 5),
    "Sad & Emotional": (0.88, -2),
    "Deep Cinematic Documentary": (0.92, -5),
    "Sarcastic & Playful Roast": (1.06, 3),
    "Motivational Goosebumps": (1.0, -2),
    "Calm Podcast Host": (0.96, -1),
    "News Anchor Crisp": (1.05, 1),
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
    box = {}
    t = threading.Thread(target=lambda: box.setdefault("r", asyncio.run(coro))); t.start(); t.join()
    return box.get("r")

def _load_kokoro():
    from kokoro import KPipeline
    return KPipeline(lang_code="a")

def tts(text, voice, style, speed, pitch, pause, strip_cues, notes):
    try:
        from pydub import AudioSegment
        if not (text or "").strip():
            return None, "❌ Please enter text."
        if strip_cues:
            text = re.sub(r"\[.*?\]", "", text)
            text = re.sub(r"\((?:breath|pause|sigh|whisper|beat)\)", "...", text, flags=re.I)
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

# ------------------------------------------------------------------ Music & Songs
MUSICGEN_MODELS = ["facebook/musicgen-medium", "facebook/musicgen-large"]
BARK_MODELS = ["suno/bark", "suno/bark-small"]

def _load_musicgen(model_id="facebook/musicgen-medium"):
    _ensure_model_from_drive(model_id)
    from transformers import AutoProcessor, MusicgenForConditionalGeneration
    proc = AutoProcessor.from_pretrained(model_id)
    model = MusicgenForConditionalGeneration.from_pretrained(model_id, torch_dtype=torch.float16).to("cuda" if torch.cuda.is_available() else "cpu")
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

def music_instrumental(prompt, my_idea, mood, duration, mg_model="facebook/musicgen-medium"):
    try:
        combined = f"{prompt}. {my_idea}".strip(". ")
        out = _make_bgm(combined, mood, duration, f"bgm_{int(duration)}s", mg_model=mg_model)
        return save_and_sync(out, "music")
    except Exception:
        return None, _err()

def _make_bgm(prompt, mood, duration, prefix, mg_model="facebook/musicgen-medium"):
    from pydub import AudioSegment
    proc, model = _use("musicgen:" + mg_model, lambda: _load_musicgen(mg_model))
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

BARK_SINGERS = {
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
GENRES = [
    "Bollywood Romantic Acoustic", "Late-Night Lo-Fi Chill", "Sad Heartbreak Piano & Strings",
    "Desi Hip-Hop / Trap Beat", "Punjabi Urban Pop", "Sufi Soulful Qawwali", "Dark Horror / Suspense Score",
    "Cinematic Epic Documentary", "Phonk / Gym Bass", "Tropical House / Influencer Vlog",
    "Classical Bansuri & Tabla", "R&B / Sensual Slow Jam", "Ghazal Harmonium & Tabla", "Indie Acoustic Guitar",
    "EDM Festival Drop", "Synthwave Retro 80s", "Jazz Lounge", "Garba / Dandiya Folk", "Bhangra Dhol Party",
    "Ambient Meditation", "Rock Anthem", "Kids Happy Ukulele", "Corporate Upbeat", "Afrobeat Groove",
    "Wedding Shehnai Celebration", "Custom Genre",
]

def _load_bark(model_id="suno/bark"):
    _ensure_model_from_drive(model_id)
    from transformers import AutoProcessor, BarkModel
    proc = AutoProcessor.from_pretrained(model_id)
    model = BarkModel.from_pretrained(model_id, torch_dtype=torch.float16).to("cuda" if torch.cuda.is_available() else "cpu")
    return proc, model

def music_song(lyrics, my_song_idea, singer, genre, mood, duration, bark_model="suno/bark", vocal_boost=3):
    try:
        from pydub import AudioSegment
        if not (lyrics or "").strip():
            return None, "❌ Please write or paste some song lyrics first."
        proc, model = _use("bark:" + bark_model, lambda: _load_bark(bark_model))
        sr = model.generation_config.sample_rate
        lines = [l.strip() for l in lyrics.splitlines() if l.strip() and not re.match(r"^\[.*\]$", l.strip())]
        song = AudioSegment.silent(0)
        tmp = os.path.join(HUB, SUBFOLDERS["music"], "_bark.wav")
        style_tag = f"{BARK_SINGERS[singer][1]}, {genre}, {mood}, {my_song_idea}".strip(", ")
        for i in range(0, len(lines), 2):
            chunk = " ".join(lines[i:i + 2])
            inputs = proc(f"♪ [{style_tag}] {chunk} ♪", voice_preset=BARK_SINGERS[singer][0]).to(model.device)
            with torch.no_grad():
                audio = model.generate(**inputs, do_sample=True).float().cpu().numpy().squeeze()
            sf.write(tmp, audio, sr)
            song = song.append(AudioSegment.from_file(tmp), crossfade=150) if len(song) else AudioSegment.from_file(tmp)
            if len(song) / 1000 >= duration:
                break
        if os.path.exists(tmp):
            os.remove(tmp)
        vocals = song[: int(duration * 1000)]
        bgm_prompt = f"{genre} instrumental backing track, {my_song_idea}, no vocals".strip(", ")
        bgm_path = _make_bgm(bgm_prompt, mood or "", max(15, duration), "_backing", mg_model="facebook/musicgen-medium")
        backing = AudioSegment.from_file(bgm_path) - 6
        os.remove(bgm_path)
        song = backing[: int(duration * 1000)].overlay(vocals + int(vocal_boost), position=1200).fade_out(2000)
        out = os.path.join(HUB, SUBFOLDERS["music"], f"song_{_ts()}.wav")
        song.export(out, format="wav")
        return save_and_sync(out, "music")
    except Exception:
        return None, _err()

# ------------------------------------------------------------------ SFX
SFX = {
    "Horror Night Creaks & Whispers": "dark eerie horror night ambience, wooden floor creaks, distant whispers",
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
    "Custom SFX": "",
}

def _load_audioldm():
    _ensure_model_from_drive("cvssp/audioldm2")
    from diffusers import AudioLDM2Pipeline
    p = AudioLDM2Pipeline.from_pretrained("cvssp/audioldm2", torch_dtype=torch.float16)
    return p.to("cuda" if torch.cuda.is_available() else "cpu")

def sfx(preset, my_sfx_idea, duration, steps=150):
    try:
        base = SFX.get(preset, "")
        prompt = f"{my_sfx_idea}. {base}".strip(". ") if (my_sfx_idea or "").strip() and preset != "Custom SFX" else ((my_sfx_idea or "").strip() or base)
        if not prompt:
            return None, "❌ Enter your custom SFX idea or select a preset."
        pipe = _use("audioldm2", _load_audioldm)
        audio = pipe(
            prompt,
            negative_prompt="low quality, distorted, noise, muffled",
            num_inference_steps=int(steps),
            audio_length_in_s=float(duration),
        ).audios[0]
        out = os.path.join(HUB, SUBFOLDERS["music"], f"sfx_{_ts()}.wav"); sf.write(out, audio, 16000)
        return save_and_sync(out, "music")
    except Exception:
        return None, _err()

# ------------------------------------------------------------------ SEO
def seo(topic, my_seo_idea, platform, custom, lang="Conversational Hinglish (Roman Hindi)", model_id=LLM_MODELS[0]):
    try:
        if not (topic or "").strip() and not (my_seo_idea or "").strip():
            return "", None, "❌ Enter your video topic, script, or custom idea."
        user = (
            f"Platform: {platform}\n"
            f"Video Topic / Script:\n{(topic or '')[:6000]}\n"
            f"My Own Angle / Target Audience / CTA Idea: {my_seo_idea or 'none'}\n"
            f"Extra SEO Instructions: {custom or 'none'}\n\n"
            "Give, in this order with clear headings:\n"
            "1. 5 scroll-stopping high-CTR hooks / titles\n"
            "2. 3 bold human influencer captions (no emoji spam, no cringe)\n"
            "3. 1 pinned comment bait that triggers replies\n"
            "4. 20 targeted hashtags (balanced mix of niche and viral broad tags on one line)\n"
            + LANG_LOCK.get(lang, "")
        )
        text = _clean(llm([{"role": "system", "content": _system(lang, True).replace("Output only the script.", "")},
                           {"role": "user", "content": user}], 1100, 0.85, model_id=model_id))
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
                if "99_model_cache" in root:
                    continue
                for f in files:
                    p = os.path.join(root, f)
                    z.write(p, os.path.relpath(p, os.path.dirname(HUB)))
        return out, f"📦 Project ZIP ready (all scripts, voices, music, SFX & SEO): {out}"
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
    kw = {"title": "AI Studio 01 — Brain & Voice (Ultra Edition)"}
    import inspect
    if "css" in inspect.signature(gr.Blocks.__init__).parameters:
        kw.update(theme=theme, css=CSS)
    with gr.Blocks(**kw) as demo:
        with gr.Row(elem_id="hdr"):
            gr.Markdown("# 🧠🎙️ AI Studio 01 — Brain & Voice (Ultra Edition)\nUnrestricted human scripts, songs, voiceovers, music, SFX & SEO — 100% free, zero-error disk & VRAM engine.")
            with gr.Column(scale=0, min_width=320):
                vram_btn = gr.Button("🧹 Clear VRAM & Free Temp Disk", variant="secondary")
                gpu_box = gr.Textbox(value=gpu_status(), show_label=False, interactive=False)
        banner = gr.Textbox(value=drive_status(), show_label=False, interactive=False, elem_id="banner")
        vram_btn.click(clear_vram_and_disk, None, gpu_box)

        with gr.Accordion("☁️ Google Drive Live Sync & 1-Time Disk-Safe Model Cache Setup", open=False):
            secrets = gr.File(label="client_secrets.json (auto-detected from /kaggle/input if attached)", file_types=[".json"])
            s1 = gr.Button("Step 1: Generate Google Sign-In Link", variant="primary")
            link = gr.Markdown()
            code = gr.Textbox(label="Step 2: Paste Verification Code")
            s2 = gr.Button("Connect & Sync Drive Now (2-Second Connect)", variant="primary")
            dmsg = gr.Textbox(label="Drive status", interactive=False)
            s1.click(drive_step1, secrets, link)
            s2.click(drive_step2, code, [dmsg, banner])

            model_picker = gr.CheckboxGroup(
                choices=ALL_MODELS,
                value=DEFAULT_CACHE_MODELS,
                label="Select Models to Cache 1-by-1 in Google Drive (Disk-Safe: downloads 1 -> uploads -> frees Kaggle disk)",
            )
            with gr.Row():
                dl_btn = gr.Button("⬇️ Cache Selected Models to Drive (1-Time Setup)", variant="primary")
                cs_btn = gr.Button("🔄 Check Cache Sync & Disk Space", variant="secondary")
            dl_msg = gr.Textbox(label="Model cache status", interactive=False, lines=6)
            dl_btn.click(download_selected_models, model_picker, dl_msg)
            cs_btn.click(cache_sync_status, None, dl_msg)

        with gr.Tabs():
            with gr.Tab("✍️ Ultra Human Writer (Scripts, Songs & Voiceovers)"):
                with gr.Row():
                    w_mode = gr.Dropdown(WRITER_MODES, value=WRITER_MODES[0], label="Studio Writer Mode (Script / Song / Voiceover / Story / Custom)")
                    llm_sel = gr.Dropdown(LLM_MODELS, value=LLM_MODELS[0], label="AI Brain Model (Hermes-3 / Mistral-Nemo-12B / Qwen-14B)")
                    bold = gr.Checkbox(value=True, label="🔥 Bold / Unfiltered / Uncensored Mode")
                with gr.Row():
                    topic = gr.Textbox(label="Topic / Song Title / Hook", lines=2, placeholder="e.g. Late night heartbreak song, or Bold AI influencer reel about midnight cravings...")
                    plat = gr.Dropdown(PLATFORMS, value=PLATFORMS[0], label="Platform / Genre Sub-Style")
                my_idea = gr.Textbox(
                    label="💡 Apna Khud Ka Idea / Storyline / Lyrics Theme / Rough Notes (Yahan apna poora idea likhein)",
                    lines=4,
                    placeholder="Apni kahani, gaane ki situation, lines, ya voiceover ka exact angle yahan khul kar likhein...",
                )
                with gr.Row():
                    lang = gr.Dropdown(LANGS, value="Conversational Hinglish (Roman Hindi)", label="Language (Hard Language Lock)")
                    tone = gr.Dropdown(TONES, value=TONES[0], label="Emotion & Vibe")
                    length = gr.Dropdown(list(LENGTHS), value="Medium (~800 words / 3-5 mins)", label="Length Preset")
                custom = gr.Textbox(
                    label="🎬 Custom Director / Song Structure / Voiceover Instructions",
                    lines=2,
                    placeholder="e.g. [Verse 1] -> [Chorus] -> [Rap] structure, rhyme scheme, character names, pauses, slang...",
                )
                with gr.Row():
                    wbtn = gr.Button("✍️ Generate Content", variant="primary")
                    ebtn = gr.Button("➕ Continue / Extend (+1000 Words)", variant="secondary")
                    hbtn = gr.Button("✨ Humanize & De-AI Polish", variant="secondary")

                script_kw = {"label": "Generated Output (Script / Song Lyrics / Voiceover)", "lines": 22}
                if "show_copy_button" in inspect.signature(gr.Textbox.__init__).parameters:
                    script_kw["show_copy_button"] = True
                script = gr.Textbox(**script_kw)

                with gr.Row():
                    to_voice_btn = gr.Button("🎙️ Send to Text-to-Voice", variant="secondary")
                    to_song_btn = gr.Button("🎤 Send to Song Singer", variant="secondary")
                    to_seo_btn = gr.Button("🚀 Send to Titles & Hashtags", variant="secondary")

                sfile = gr.File(label="Saved .txt"); smsg = gr.Textbox(label="Status", interactive=False)
                wbtn.click(write_script, [w_mode, topic, my_idea, plat, lang, tone, length, custom, llm_sel, bold], [script, sfile, smsg])
                ebtn.click(extend_script, [script, w_mode, lang, tone, my_idea, custom, llm_sel, bold], [script, sfile, smsg])
                hbtn.click(humanize_polish, [script, lang, custom, llm_sel, bold], [script, sfile, smsg])

            with gr.Tab("🎙️ Text-to-Voice"):
                ttext = gr.Textbox(label="Voiceover Text (blank line = new paragraph / breath pause)", lines=10)
                with gr.Row():
                    voice = gr.Dropdown(list(VOICES), value=list(VOICES)[0], label="Voice")
                    style = gr.Dropdown(list(STYLES), value="Natural Conversational", label="Speaking Style Preset")
                    strip_cues = gr.Checkbox(value=True, label="Auto-clean [VISUAL SCENE] tags before speaking")
                with gr.Row():
                    speed = gr.Slider(0.65, 1.40, value=0.95, step=0.01, label="Speed")
                    pitch = gr.Slider(-25, 25, value=0, step=1, label="Pitch (Hz)")
                    pause = gr.Slider(0.0, 1.5, value=0.30, step=0.05, label="Breath Pause Between Paragraphs (s)")
                notes = gr.Textbox(label="Custom Pronunciation / Pause Notes (one per line: word = spoken form)", lines=2)
                tbtn = gr.Button("🎙️ Generate Voice", variant="primary")
                taudio = gr.Audio(label="Voice", type="filepath"); tmsg = gr.Textbox(label="Status", interactive=False)
                tbtn.click(tts, [ttext, voice, style, speed, pitch, pause, strip_cues, notes], [taudio, tmsg])

            with gr.Tab("🎵 Music & Songs"):
                with gr.Tab("Mode A — Long Instrumental / BGM"):
                    with gr.Row():
                        mg_model = gr.Dropdown(MUSICGEN_MODELS, value=MUSICGEN_MODELS[0], label="MusicGen Model")
                        mdur = gr.Slider(15, 300, value=60, step=5, label="Duration (s)")
                    mprompt = gr.Textbox(label="Music Style / Instruments", value="emotional cinematic piano with soft strings and warm bass")
                    my_bgm_idea = gr.Textbox(label="💡 Apna Khud Ka Music Idea / Scene Vibe", lines=2, placeholder="e.g. Late night rainy drive in Mumbai, slow build-up with emotional drop at 30s...")
                    mmood = gr.Textbox(label="Custom Mood / Tempo")
                    mbtn = gr.Button("🎵 Generate BGM", variant="primary")
                    maudio = gr.Audio(label="BGM", type="filepath"); mmsg = gr.Textbox(label="Status", interactive=False)
                    mbtn.click(music_instrumental, [mprompt, my_bgm_idea, mmood, mdur, mg_model], [maudio, mmsg])

                with gr.Tab("Mode B — Full Song with Vocals"):
                    lyr = gr.Textbox(label="Song Lyrics ([Verse], [Chorus], ♪)", lines=10)
                    my_song_idea = gr.Textbox(label="💡 Apna Khud Ka Song Direction / Melody & Beat Idea", lines=2, placeholder="e.g. Arijit Singh style emotional slow start, heavy tabla & acoustic guitar in chorus...")
                    with gr.Row():
                        singer = gr.Dropdown(list(BARK_SINGERS), value=list(BARK_SINGERS)[0], label="Singer Voice")
                        genre = gr.Dropdown(GENRES, value=GENRES[0], label="Genre (or type custom)", allow_custom_value=True)
                        bark_sel = gr.Dropdown(BARK_MODELS, value=BARK_MODELS[0], label="Vocal Engine (Bark Full = richer emotion)")
                    with gr.Row():
                        smood = gr.Textbox(label="Custom Mood")
                        sdur = gr.Slider(15, 300, value=45, step=5, label="Duration (s)")
                        vboost = gr.Slider(-3, 8, value=3, step=1, label="Vocal vs Music Balance (dB)")
                    sbtn = gr.Button("🎤 Generate Full Song", variant="primary")
                    saudio = gr.Audio(label="Song", type="filepath"); smsg2 = gr.Textbox(label="Status", interactive=False)
                    sbtn.click(music_song, [lyr, my_song_idea, singer, genre, smood, sdur, bark_sel, vboost], [saudio, smsg2])

            with gr.Tab("🔊 Sound Effects"):
                with gr.Row():
                    fpre = gr.Dropdown(list(SFX), value=list(SFX)[0], label="SFX Preset")
                    fdur = gr.Slider(2, 20, value=6, step=1, label="Duration (s)")
                    fsteps = gr.Slider(50, 200, value=150, step=10, label="Quality Steps")
                my_sfx_idea = gr.Textbox(label="💡 Apna Khud Ka SFX Idea / Custom Scene Sound", lines=2, placeholder="e.g. Heavy Indian monsoon rain hitting a tin roof with distant temple bells and thunder...")
                fbtn = gr.Button("🔊 Generate SFX", variant="primary")
                faudio = gr.Audio(label="SFX", type="filepath"); fmsg = gr.Textbox(label="Status", interactive=False)
                fbtn.click(sfx, [fpre, my_sfx_idea, fdur, fsteps], [faudio, fmsg])

            with gr.Tab("🚀 Titles & Hashtags"):
                qtopic = gr.Textbox(label="Video topic or paste your script / lyrics", lines=6)
                my_seo_idea = gr.Textbox(label="💡 Apna Khud Ka Angle / Target Audience / Hook Idea", lines=2, placeholder="e.g. Target Hindi youth audience, curiosity hook, ask viewers to comment their story...")
                with gr.Row():
                    qplat = gr.Dropdown(["Instagram Reels", "YouTube Long Video", "YouTube Shorts", "Facebook / X"], value="Instagram Reels", label="Platform")
                    qlang = gr.Dropdown(LANGS, value="Conversational Hinglish (Roman Hindi)", label="Language")
                    qllm = gr.Dropdown(LLM_MODELS, value=LLM_MODELS[0], label="Model")
                qcus = gr.Textbox(label="Custom SEO Instructions")
                qbtn = gr.Button("🚀 Generate SEO Pack", variant="primary")
                qout = gr.Textbox(label="SEO Pack", lines=20); qfile = gr.File(label="Saved .txt"); qmsg = gr.Textbox(label="Status", interactive=False)
                qbtn.click(seo, [qtopic, my_seo_idea, qplat, qcus, qlang, qllm], [qout, qfile, qmsg])

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

        # Wire cross-tab quick-send buttons
        to_voice_btn.click(send_to_voice, script, [ttext, smsg])
        to_song_btn.click(send_to_song, script, [lyr, smsg])
        to_seo_btn.click(send_to_seo, script, [qtopic, smsg])

    _orig = demo.launch
    def launch(*a, **k):
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

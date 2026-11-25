"""Studio_01_Brain_Voice_Audio v3 - Kaggle (GPU T4 x2, Internet ON, Python 3.13). Zero paid APIs.
Unrestricted scripts + SEO (Hermes-3-Llama-3.1-8B, fallback Qwen2.5-14B, 4-bit), voices (edge-tts + Kokoro),
long music + songs (MusicGen + Bark), SFX (AudioLDM2), splitter (Demucs), in-UI Google Drive connect."""
import os, sys, glob, shutil
from datetime import datetime
from urllib.parse import urlparse, parse_qs

HUB_NAME = "AI_Studio_Master_Hub"
LOCAL_HUB = f"/kaggle/working/{HUB_NAME}"
SUBFOLDERS = ["00_reference_faces", "01_scripts", "02_voices", "03_images",
              "04_raw_videos", "05_music", "06_captions_hashtags", "07_final_exports"]
FOLDER_MIME = "application/vnd.google-apps.folder"

drive = None          # PyDrive2 GoogleDrive instance (None = local-only mode)
drive_ids = {}        # subfolder name -> Drive folder id



# ---------------- Google Drive (PyDrive2) - connect from inside the Gradio UI ----------------
CREDS_FILE = "/kaggle/working/drive_creds.json"
_auth = {"gauth": None}

def _find_client_secrets():
    for p in ["/kaggle/working/client_secrets.json", "client_secrets.json",
              *glob.glob("/kaggle/input/**/client_secrets*.json", recursive=True)]:
        if os.path.exists(p):
            return p
    return None

def _new_gauth(secrets):
    from pydrive2.auth import GoogleAuth
    return GoogleAuth(settings={
        "client_config_backend": "file", "client_config_file": secrets,
        "save_credentials": True, "save_credentials_backend": "file", "save_credentials_file": CREDS_FILE,
        "get_refresh_token": True, "oauth_scope": ["https://www.googleapis.com/auth/drive"],
    })

def _finish(gauth):
    global drive
    from pydrive2.drive import GoogleDrive
    gauth.SaveCredentialsFile(CREDS_FILE)
    drive = GoogleDrive(gauth)
    setup_folders()

def badge():
    return (f"### ✅ Google Drive Sync: LIVE CONNECTED ({HUB_NAME})" if drive is not None
            else "### Drive Sync: Local Mode")

def try_saved_login():
    """Silently reuses a saved login (drive_creds.json) if one exists. Never asks for input."""
    secrets = _find_client_secrets()
    if not secrets or not os.path.exists(CREDS_FILE):
        return False
    try:
        g = _new_gauth(secrets); g.LoadCredentialsFile(CREDS_FILE)
        if g.credentials is None: return False
        g.Refresh() if g.access_token_expired else g.Authorize()
        _finish(g); return True
    except Exception as e:
        print("Saved Drive login could not be reused:", e); return False

def ui_step1(upload):
    if upload is not None:
        src = upload if isinstance(upload, str) else upload.name
        shutil.copy(src, "/kaggle/working/client_secrets.json")
    secrets = _find_client_secrets()
    if not secrets:
        return "Please upload client_secrets.json first (see README > Google Drive setup).", badge()
    g = _new_gauth(secrets); g.GetFlow()
    g.flow.redirect_uri = "http://localhost:8080/"
    _auth["gauth"] = g
    url = g.GetAuthUrl()
    return (f"**1.** [Click here to sign in with Google]({url})\n\n"
            "**2.** After you allow access the page says *site can't be reached* - that is normal. "
            "Copy the full address from the address bar (or just the code) and paste it in Step 2."), badge()

def ui_connect(raw):
    g = _auth["gauth"]
    if g is None:
        return "Click 'Step 1: Generate Google Login Link' first.", badge()
    raw = (raw or "").strip()
    if not raw:
        return "Paste the Google code or the full address first.", badge()
    code = parse_qs(urlparse(raw).query).get("code", [raw])[0] if raw.startswith("http") else raw
    try:
        g.Auth(code); _finish(g)
        return f"Connected. Folder My Drive/{HUB_NAME} is ready and new files will upload automatically.", badge()
    except Exception as e:
        return f"Login failed: {e}. Generate a new link and try again.", badge()

def _get_or_create_folder(name, parent_id):
    """Reuses an existing folder; creates only if missing. Never deletes or renames."""
    q = f"title='{name}' and mimeType='{FOLDER_MIME}' and '{parent_id}' in parents and trashed=false"
    found = drive.ListFile({"q": q}).GetList()
    if found:
        return found[0]["id"]
    f = drive.CreateFile({"title": name, "mimeType": FOLDER_MIME, "parents": [{"id": parent_id}]})
    f.Upload()
    return f["id"]


def setup_folders():
    for s in SUBFOLDERS:
        os.makedirs(os.path.join(LOCAL_HUB, s), exist_ok=True)
    if drive is not None:
        root = _get_or_create_folder(HUB_NAME, "root")
        for s in SUBFOLDERS:
            drive_ids[s] = _get_or_create_folder(s, root)
        print(f"Drive folder My Drive/{HUB_NAME} ready with {len(SUBFOLDERS)} subfolders.")
    print("Local folder ready:", LOCAL_HUB)


def save_and_sync_to_drive(local_file_path, subfolder_name):
    """File is already saved locally under LOCAL_HUB/subfolder; uploads a copy into the matching Drive folder."""
    if drive is None or subfolder_name not in drive_ids:
        return f"Saved locally only: {local_file_path}"
    try:
        f = drive.CreateFile({"title": os.path.basename(local_file_path),
                              "parents": [{"id": drive_ids[subfolder_name]}]})
        f.SetContentFile(local_file_path)
        f.Upload()
        return f"Synced to Drive: {HUB_NAME}/{subfolder_name}/{os.path.basename(local_file_path)}"
    except Exception as e:
        return f"Saved locally, Drive upload failed: {e}"


def save_and_sync(local_path, subfolder):
    return save_and_sync_to_drive(local_path, subfolder)


def _out(subfolder, prefix, ext):
    folder = os.path.join(LOCAL_HUB, subfolder); os.makedirs(folder, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    p, n = os.path.join(folder, f"{prefix}_{stamp}.{ext}"), 1
    while os.path.exists(p):  # never overwrite
        p = os.path.join(folder, f"{prefix}_{stamp}_{n}.{ext}"); n += 1
    return p


# ---------------- VRAM auto-cleaner: only one model in memory at a time ----------------
_model = {"name": None, "obj": None}

def _load(name, loader):
    import gc, torch
    if _model["name"] != name:
        _model["obj"] = None; _model["name"] = None
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        _model["obj"] = loader(); _model["name"] = name
    return _model["obj"]


def _free():
    import gc, torch
    _model["obj"] = None; _model["name"] = None
    gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()



# ---------------- LLM: Hermes-3 8B (fallback Qwen2.5-14B), 4-bit (Tab 1 + Tab 5) ----------------
BANNED = ["delve", "tapestry", "testament", "in conclusion", "ever-evolving", "embark", "realm",
          "unleash", "game-changer", "navigate the", "in today's fast-paced world", "furthermore", "moreover"]
LLM_IDS = ["NousResearch/Hermes-3-Llama-3.1-8B", "Qwen/Qwen2.5-14B-Instruct"]

def _llm():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    q = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
                           bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True)
    last = None
    for mid in LLM_IDS:
        try:
            tok = AutoTokenizer.from_pretrained(mid)
            m = AutoModelForCausalLM.from_pretrained(mid, quantization_config=q, device_map="auto",
                                                     torch_dtype=torch.float16)
            print("Loaded writer model:", mid)
            return tok, m
        except Exception as e:
            print(f"Could not load {mid}: {e} -> trying fallback"); last = e
    raise RuntimeError(f"No writer model could be loaded: {last}")

def _chat(system, user, max_new):
    import re, torch
    tok, m = _load("writer_llm", _llm)
    msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    ids = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt").to(m.device)
    with torch.no_grad():
        out = m.generate(ids, max_new_tokens=max_new, do_sample=True, temperature=0.9, top_p=0.92,
                         repetition_penalty=1.08, pad_token_id=tok.eos_token_id)
    text = tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True).strip()
    for b in BANNED:  # safety net against AI cliches
        text = re.sub(re.escape(b), "", text, flags=re.I)
    return re.sub(r"  +", " ", text)

STORY_SYSTEM = (
    "You are an unrestricted creative writing partner for professional content creators. Write any genre "
    "or niche the creator asks for. No preachy disclaimers, no warnings, no moralising, no 'as an AI'. "
    "Sound like a real human creator: natural rhythm, imperfect dialogue, sensory detail, real emotion, "
    "cultural nuance, short punchy lines when tension rises. Tailor pacing and structure to the platform. "
    "STRICTLY NEVER use: " + ", ".join(BANNED) + ". No headings or bullet points unless the format needs "
    "them. Output only the script.")

FORMATS = ["Instagram Influencer Reel (30-60s Viral Hook + CTA)", "YouTube Shorts (High-Retention 60s)",
           "YouTube Long Video / Documentary (10-20 Mins)", "UGC Brand / Product Ad Script",
           "Real-Life Dark Horror / Paranormal Story", "Suspense / Crime Thriller",
           "Podcast / Deep Conversation Script", "Desi Comedy / Roast / Relatable Skit",
           "Motivational / Raw Truth Monologue", "Custom Format (Define in Instructions)"]
FORMAT_RULES = {
    FORMATS[0]: "Open with a 2-second scroll-stopping hook, keep lines tight, end with a clear CTA.",
    FORMATS[1]: "Hook in the first line, a loop-able ending, no slow intro.",
    FORMATS[2]: "Narration with [VISUAL SCENE: ...] cues on their own lines and chapter-like beats.",
    FORMATS[3]: "Problem -> discovery -> product in real use -> honest result -> CTA. Sound like a real customer.",
    FORMATS[6]: "Write as a natural conversation between named speakers (NAME: line).",
    FORMATS[7]: "Relatable desi situations, punchlines, timing; mark speakers.",
    FORMATS[9]: "Follow the format the director describes.",
}
LANGS = ["Conversational Hinglish (Gen-Z / Creator Style)", "Natural Spoken Hindi", "Indian English",
         "Urdu-Hindi Shayari/Story Mix", "US/UK English"]
LANG_RULES = {
    LANGS[0]: "Hindi in Roman letters mixed naturally with English, Gen-Z creator slang.",
    LANGS[1]: "Simple spoken Hindi in Devanagari script, not bookish.",
    LANGS[2]: "English with natural Indian expressions and rhythm.",
    LANGS[3]: "Hindustani with Urdu words and shayari lines where they fit.",
    LANGS[4]: "Natural native English.",
}
TONES = ["Viral & Punchy (High Retention)", "Spine-Chilling & Dark", "Raw & Emotional",
         "Bold / Unfiltered / Edgy", "Cinematic Documentary", "Funny & Sarcastic"]
LENGTHS = {"Short (150-300 words)": 250, "Medium (600-1000 words)": 850, "Long-Form (1500-2500 words)": 2200}

def write_script(topic, fmt, language, tone, length, director):
    if not topic.strip():
        return "Please enter a topic, hook or idea.", None, "", ""
    words = LENGTHS[length]
    user = (f"Topic / hook: {topic}\nFormat: {fmt} - {FORMAT_RULES.get(fmt, '')}\n"
            f"Language: {language} - {LANG_RULES.get(language, '')}\nTone: {tone}\n"
            f"Target length: about {words} words.\n")
    if director and director.strip():
        user += f"Director's instructions (follow exactly): {director.strip()}\n"
    text = _chat(STORY_SYSTEM, user, max_new=min(4096, int(words * 2.2) + 200))
    p = _out("01_scripts", "script", "txt")
    with open(p, "w", encoding="utf-8") as f: f.write(text)
    return text, p, save_and_sync(p, "01_scripts"), p

def extend_script(current, fmt, language, tone, director, path):
    if not (current or "").strip():
        return current, None, "Write a script first, then extend it.", path
    tail = current[-6000:]
    user = (f"Here is the script so far (format: {fmt}, language: {language}, tone: {tone}):\n\n{tail}\n\n"
            "Continue it seamlessly from the last line for about 1000 more words. Same characters, voice and "
            "language. Do not repeat or summarise earlier parts. Output only the continuation.")
    if director and director.strip():
        user += f"\nDirector's instructions: {director.strip()}"
    more = _chat(STORY_SYSTEM, user, max_new=2400)
    full = current.rstrip() + "\n\n" + more
    p = path if path and os.path.exists(path) else _out("01_scripts", "script", "txt")
    ext = _out("01_scripts", "script_extended", "txt")  # new file each time, never overwrite
    with open(ext, "w", encoding="utf-8") as f: f.write(full)
    return full, ext, save_and_sync(ext, "01_scripts"), ext


# ---------------- Tab 5: Titles, hooks, captions, hashtags ----------------
SEO_SYSTEM = ("You are a viral social media strategist who writes like a real influencer. No cliches, "
              "no emoji spam, no disclaimers. NEVER use: " + ", ".join(BANNED) + ".")
PLATFORMS = ["Instagram Reels", "YouTube Long", "YouTube Shorts", "LinkedIn/X"]

def seo(topic, platform, language, custom):
    if not topic.strip():
        return "Please enter the video topic or paste the script.", None, ""
    user = (f"Content: {topic}\nPlatform: {platform}\nLanguage: {language}\n"
            "Give exactly these sections:\n"
            "HOOKS / TITLES: 5 scroll-stopping options (numbered)\n"
            "CAPTIONS: 3 human influencer-style captions (numbered)\n"
            "PINNED COMMENT: 1 comment-bait line that makes people reply\n"
            "HASHTAGS: 20 targeted hashtags mixing niche and broad, on one line\n")
    if custom and custom.strip():
        user += f"Extra instructions: {custom.strip()}\n"
    text = _chat(SEO_SYSTEM, user, max_new=1000)
    p = _out("06_captions_hashtags", "seo", "txt")
    with open(p, "w", encoding="utf-8") as f: f.write(text)
    return text, p, save_and_sync(p, "06_captions_hashtags")

# ---------------- Tab 2: Text to voice (edge-tts + Kokoro) ----------------
VOICES = {
    "Hindi - Swara (Female, India)": ("edge", "hi-IN-SwaraNeural"),
    "Hindi - Madhur (Male, India)": ("edge", "hi-IN-MadhurNeural"),
    "Hinglish - Swara (Female, India)": ("edge", "hi-IN-SwaraNeural"),
    "Indian English - Neerja (Female)": ("edge", "en-IN-NeerjaNeural"),
    "Indian English - Prabhat (Male)": ("edge", "en-IN-PrabhatNeural"),
    "Urdu - Uzma (Female, Pakistan)": ("edge", "ur-PK-UzmaNeural"),
    "Urdu - Asad (Male, Pakistan)": ("edge", "ur-PK-AsadNeural"),
    "Bengali - Tanishaa (Female, India)": ("edge", "bn-IN-TanishaaNeural"),
    "Bengali - Bashkar (Male, India)": ("edge", "bn-IN-BashkarNeural"),
    "Marathi - Aarohi (Female, India)": ("edge", "mr-IN-AarohiNeural"),
    "Marathi - Manohar (Male, India)": ("edge", "mr-IN-ManoharNeural"),
    "Tamil - Pallavi (Female, India)": ("edge", "ta-IN-PallaviNeural"),
    "Tamil - Valluvar (Male, India)": ("edge", "ta-IN-ValluvarNeural"),
    "US English - Aria (Female, Global)": ("edge", "en-US-AriaNeural"),
    "US English - Guy (Male, Global)": ("edge", "en-US-GuyNeural"),
    "UK English - Sonia (Female, Global)": ("edge", "en-GB-SoniaNeural"),
    "Kokoro - Heart (US Female, offline)": ("kokoro", ("a", "af_heart")),
    "Kokoro - Michael (US Male, offline)": ("kokoro", ("a", "am_michael")),
    "Kokoro - Emma (UK Female, offline)": ("kokoro", ("b", "bf_emma")),
    "Kokoro - George (UK Male, offline)": ("kokoro", ("b", "bm_george")),
    "Kokoro - Alpha (Hindi Female, offline)": ("kokoro", ("h", "hf_alpha")),
    "Kokoro - Omega (Hindi Male, offline)": ("kokoro", ("h", "hm_omega")),
}
STYLES = {  # (speed multiplier, pitch offset Hz)
    "Influencer Energetic": (1.12, 4),
    "Dark Horror Whisper": (0.88, -10),
    "Deep Documentary": (0.94, -5),
    "Emotional Soft": (0.90, -2),
    "Natural Chat": (1.00, 0),
}

def _apply_pronunciation(text, notes):
    """Notes like 'Kaggle=Kaa-gel, AI=A I' (one per line or comma) replace words before speaking."""
    import re
    for part in re.split(r"[,\n]", notes or ""):
        if "=" in part:
            a, b = [x.strip() for x in part.split("=", 1)]
            if a: text = re.sub(rf"\b{re.escape(a)}\b", b, text)
    return text

def _edge_tts(text, voice, rate_pct, pitch_hz, path):
    import asyncio, edge_tts
    async def run():
        c = edge_tts.Communicate(text, voice, rate=f"{rate_pct:+d}%", pitch=f"{pitch_hz:+d}Hz")
        await c.save(path)
    try:
        asyncio.run(run())
    except RuntimeError:  # an event loop is already running
        import threading
        t = threading.Thread(target=lambda: asyncio.run(run())); t.start(); t.join()


def tts(text, voice_label, style, speed, pitch, pause, notes):
    from pydub import AudioSegment
    if not text.strip():
        return None, None, "Please enter the script text."
    engine, voice = VOICES[voice_label]
    s_mul, s_pitch = STYLES[style]
    spd = max(0.5, float(speed) * s_mul)
    pit = int(pitch) + s_pitch
    text = _apply_pronunciation(text, notes)
    paras = [p.strip() for p in text.split("\n") if p.strip()]
    tmp = os.path.join(LOCAL_HUB, "02_voices", "_tmp"); os.makedirs(tmp, exist_ok=True)
    full = AudioSegment.silent(duration=200)
    gap = AudioSegment.silent(duration=int(float(pause) * 1000))
    if engine == "edge":
        for i, para in enumerate(paras):
            pth = os.path.join(tmp, f"p{i}.mp3")
            _edge_tts(para, voice, int(round((spd - 1) * 100)), pit, pth)
            full += AudioSegment.from_file(pth) + gap
    else:
        import numpy as np, soundfile as sf
        lang, vname = voice
        pipe = _load(f"kokoro_{lang}", lambda: __import__("kokoro").KPipeline(lang_code=lang))
        for i, para in enumerate(paras):
            chunks = [a for _, _, a in pipe(para, voice=vname, speed=spd)]
            pth = os.path.join(tmp, f"p{i}.wav")
            sf.write(pth, np.concatenate([np.asarray(c) for c in chunks]), 24000)
            seg = AudioSegment.from_file(pth)
            if pit:  # simple pitch shift by resampling
                f = 2 ** (pit / 120)
                seg = seg._spawn(seg.raw_data, overrides={"frame_rate": int(seg.frame_rate * f)}).set_frame_rate(24000)
            full += seg + gap
    wav = _out("02_voices", "voice", "wav")
    full.export(wav, format="wav")
    mp3 = wav[:-4] + ".mp3"
    full.export(mp3, format="mp3", bitrate="320k")
    msg = save_and_sync(wav, "02_voices") + "\n" + save_and_sync(mp3, "02_voices")
    return wav, mp3, msg


# ---------------- Tab 3: Long music (MusicGen) + songs (Bark) ----------------
def _musicgen():
    import torch
    from transformers import AutoProcessor, MusicgenForConditionalGeneration
    proc = AutoProcessor.from_pretrained("facebook/musicgen-medium")
    m = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-medium",
                                                         torch_dtype=torch.float16).to("cuda")
    return proc, m

def _long_music(prompt, seconds):
    """Sliding-window continuation: each 30s window is conditioned on the last 10s, joined with crossfades."""
    import torch, numpy as np, soundfile as sf
    from pydub import AudioSegment
    proc, m = _load("musicgen", _musicgen)
    sr = m.config.audio_encoder.sampling_rate
    tps = m.config.audio_encoder.frame_rate  # tokens per second (50)
    WIN, CTX, XF = 30, 10, 2000
    tmp = os.path.join(LOCAL_HUB, "05_music", "_tmp"); os.makedirs(tmp, exist_ok=True)
    song, prev = None, None
    remaining = float(seconds)
    i = 0
    while remaining > 0.5:
        new_len = min(WIN if prev is None else WIN - CTX, remaining)
        with torch.no_grad():
            if prev is None:
                inp = proc(text=[prompt], padding=True, return_tensors="pt").to("cuda")
                out = m.generate(**inp, max_new_tokens=int(new_len * tps) + 4, do_sample=True, guidance_scale=3)
                audio = out[0, 0].float().cpu().numpy()
            else:
                ctx = prev[-CTX * sr:]
                inp = proc(audio=[ctx], sampling_rate=sr, text=[prompt], padding=True, return_tensors="pt").to("cuda")
                inp["input_values"] = inp["input_values"].half()
                out = m.generate(**inp, max_new_tokens=int((CTX + new_len) * tps) + 4, do_sample=True, guidance_scale=3)
                audio = out[0, 0].float().cpu().numpy()[len(ctx):]  # drop the repeated context
        prev = audio if prev is None else np.concatenate([prev, audio])
        pth = os.path.join(tmp, f"w{i}.wav"); sf.write(pth, audio, sr)
        seg = AudioSegment.from_file(pth)
        song = seg if song is None else song.append(seg, crossfade=min(XF, len(seg) // 2))
        remaining -= new_len; i += 1
    return song[: int(seconds * 1000)].fade_in(500).fade_out(1500)

def music_mode_a(prompt, seconds):
    if not prompt.strip():
        return None, "Please describe the music."
    song = _long_music(prompt, seconds)
    p = _out("05_music", "bgm", "wav"); song.export(p, format="wav")
    return p, save_and_sync(p, "05_music")

SINGERS = {
    "Male": "v2/hi_speaker_8", "Female": "v2/hi_speaker_9",
    "Male - English": "v2/en_speaker_6", "Female - English": "v2/en_speaker_9",
}
GENRES = {
    "Bollywood Acoustic": "bollywood acoustic guitar ballad, soft tabla, warm, emotional",
    "Lo-Fi": "lo-fi chill hip hop beat, vinyl crackle, mellow keys",
    "Hip-Hop Rap": "hip hop rap beat, punchy 808 drums, dark synth",
    "Sufi": "sufi folk, harmonium, dholak, spiritual, qawwali feel",
    "Sad Soulful": "sad soulful piano and strings, slow, melancholic",
    "Phonk/Trap": "phonk trap beat, cowbell, distorted 808, aggressive drift phonk",
}

def _bark():
    import torch
    from transformers import AutoProcessor, BarkModel
    proc = AutoProcessor.from_pretrained("suno/bark")
    m = BarkModel.from_pretrained("suno/bark", torch_dtype=torch.float16).to("cuda")
    return proc, m

def music_mode_b(lyrics, singer, genre, custom, seconds):
    import torch, numpy as np, soundfile as sf
    from pydub import AudioSegment
    if not lyrics.strip():
        return None, "Please write the song lyrics."
    proc, m = _load("bark", _bark)
    sr = m.generation_config.sample_rate
    tmp = os.path.join(LOCAL_HUB, "05_music", "_tmp"); os.makedirs(tmp, exist_ok=True)
    vocals = AudioSegment.silent(duration=300)
    lines = [l.strip() for l in lyrics.split("\n") if l.strip()]
    for i, line in enumerate(lines):
        if line.startswith("[") and line.endswith("]"):  # [Verse] / [Chorus] -> short breath
            vocals += AudioSegment.silent(duration=700); continue
        sung = line if "♪" in line else f"♪ {line} ♪"
        inp = proc(sung, voice_preset=SINGERS[singer]).to("cuda")
        with torch.no_grad():
            a = m.generate(**inp, do_sample=True).float().cpu().numpy().squeeze()
        pth = os.path.join(tmp, f"v{i}.wav"); sf.write(pth, a, sr)
        vocals += AudioSegment.from_file(pth) + AudioSegment.silent(duration=250)
        if len(vocals) >= seconds * 1000: break
    _free()  # unload Bark before MusicGen
    dur = max(len(vocals) / 1000 + 2, 15)
    bgm = _long_music(f"{GENRES[genre]}, {custom or ''}, instrumental backing track", min(dur, 180))
    mix = (bgm - 9).overlay(vocals + 2, position=1000)
    mix = mix[: int(max(seconds, len(vocals) / 1000 + 2) * 1000)].fade_out(1500)
    p = _out("05_music", "song", "wav"); mix.export(p, format="wav")
    return p, save_and_sync(p, "05_music")



# ---------------- Tab 4: SFX (AudioLDM2) ----------------
ENV = {
    "Horror Ambience": "eerie horror ambience, distant whispers, creaking wood, cold wind",
    "Cinematic Whoosh & Bass Drop": "cinematic whoosh followed by a deep sub bass drop impact",
    "Rain & Thunder": "heavy rain on a tin roof, rolling thunder",
    "Indian Street/Cafe": "busy Indian street cafe, chai glasses clinking, auto rickshaw horns, chatter",
    "Phone/Social Notification": "short crisp smartphone notification ping, social media alert sound",
    "Custom": "",
}

def _aldm():
    import torch
    from diffusers import AudioLDM2Pipeline
    return AudioLDM2Pipeline.from_pretrained("cvssp/audioldm2", torch_dtype=torch.float16).to("cuda")

def sfx(desc, env, seconds):
    import soundfile as sf
    prompt = ", ".join(x for x in [desc.strip(), ENV.get(env, "")] if x)
    if not prompt:
        return None, "Please describe the sound or pick a preset."
    pipe = _load("audioldm2", _aldm)
    a = pipe(prompt, negative_prompt="low quality, noise, distorted", num_inference_steps=150,
             audio_length_in_s=float(seconds)).audios[0]
    p = _out("05_music", "sfx", "wav"); sf.write(p, a, 16000)
    return p, save_and_sync(p, "05_music")


# ---------------- Tab 6: Splitter + ZIP ----------------
def split(file):
    import subprocess, shutil
    if file is None:
        return None, None, "Please upload a song or video."
    src = file if isinstance(file, str) else file.name
    _free()
    work = os.path.join(LOCAL_HUB, "05_music", "_demucs")
    r = subprocess.run([sys.executable, "-m", "demucs", "--two-stems=vocals", "-o", work, src],
                       capture_output=True, text=True)
    if r.returncode != 0:
        return None, None, "Demucs failed:\n" + r.stderr[-1500:]
    base = os.path.splitext(os.path.basename(src))[0]
    found = glob.glob(os.path.join(work, "*", base, "vocals.wav"))
    if not found:
        return None, None, "Demucs finished but output was not found."
    d = os.path.dirname(found[0])
    v, b = _out("05_music", "vocals", "wav"), _out("05_music", "bgm_only", "wav")
    shutil.copy(os.path.join(d, "vocals.wav"), v); shutil.copy(os.path.join(d, "no_vocals.wav"), b)
    return v, b, save_and_sync(v, "05_music") + "\n" + save_and_sync(b, "05_music")

def make_zip():
    import shutil
    base = f"/kaggle/working/{HUB_NAME}"
    return shutil.make_archive(base, "zip", LOCAL_HUB) if os.path.isdir(LOCAL_HUB) else None



# ---------------- UI ----------------
CSS = """
:root,.dark,body,gradio-app{
  --body-background-fill:#090b10 !important;--background-fill-primary:#090b10 !important;
  --background-fill-secondary:#111622 !important;--block-background-fill:#111622 !important;
  --input-background-fill:#182030 !important;--input-background-fill-focus:#1d2639 !important;
  --body-text-color:#f8fafc !important;--body-text-color-subdued:#cbd5e1 !important;
  --block-label-text-color:#fbbf24 !important;--block-title-text-color:#fbbf24 !important;
  --block-label-background-fill:#182030 !important;--border-color-primary:#2a3447 !important;
  --panel-background-fill:#111622 !important;--table-even-background-fill:#111622 !important;
  --table-odd-background-fill:#182030 !important;--color-accent:#f59e0b !important;
  --button-secondary-background-fill:#182030 !important;--button-secondary-text-color:#f8fafc !important;
  --button-primary-background-fill:#f59e0b !important;--button-primary-text-color:#000000 !important;
  --checkbox-background-color:#182030 !important;--neutral-50:#182030 !important;}
.gradio-container{background:#090b10 !important;color:#f8fafc !important}
.block,.form,.panel,fieldset,.gr-group{background:#111622 !important;border-color:#2a3447 !important}
textarea,input,select,.wrap,.secondary-wrap,ul.options,.file-preview,.upload-container,.empty{
  background:#182030 !important;color:#f8fafc !important;border-color:#2a3447 !important}
ul.options li{color:#f8fafc !important}
label span,.label-wrap span,span[data-testid=block-info]{color:#fbbf24 !important}
.prose,.prose *,.markdown,.markdown *,p,h1,h2,h3{color:#f8fafc !important}
.prose a{color:#fbbf24 !important;text-decoration:underline}
.tab-nav,div[role=tablist]{gap:6px;border:none !important}
.tab-nav button,button[role=tab]{color:#f8fafc !important;background:#182030 !important;border-radius:999px !important;
  border:1px solid #2a3447 !important;font-weight:600 !important;opacity:1 !important;padding:6px 16px !important}
.tab-nav button.selected,button[role=tab][aria-selected=true]{background:#f59e0b !important;color:#000000 !important;
  font-weight:800 !important;box-shadow:0 0 14px #f59e0b99 !important;border-color:#f59e0b !important}
button.primary{background:#f59e0b !important;color:#000 !important;font-weight:700 !important}
#topbar{align-items:center}
"""

def clear_vram():
    _free()
    try:
        import torch
        if torch.cuda.is_available():
            free = sum(torch.cuda.mem_get_info(i)[0] for i in range(torch.cuda.device_count())) / 1e9
            return f"GPU memory cleared. Free VRAM: {free:.1f} GB"
    except Exception:
        pass
    return "Memory cleared."

def build_ui():
    import gradio as gr
    setup_folders()
    try_saved_login()
    theme = gr.themes.Soft(primary_hue="amber", neutral_hue="slate")
    with gr.Blocks(title="Studio 01 - Brain, Voice & Audio", theme=theme, css=CSS) as ui:
        with gr.Row(elem_id="topbar"):
            gr.HTML("<h1 style='color:#fbbf24;margin:0'>Studio 01 - Brain, Voice &amp; Audio</h1>")
            vb = gr.Button("🗑️ Clear VRAM / Free GPU Memory", size="sm")
        bdg = gr.Markdown(badge())
        vs = gr.Markdown()
        vb.click(clear_vram, None, vs)
        with gr.Accordion("☁️ Google Drive Live Sync Setup (Click to Connect)", open=False):
            gr.Markdown("Upload your `client_secrets.json` (or attach it as a Kaggle dataset - it is found "
                        "automatically), then follow the two steps. Everything is always saved in "
                        f"`{LOCAL_HUB}` too.")
            cs_up = gr.File(label="client_secrets.json", file_types=[".json"])
            s1 = gr.Button("Step 1: Generate Google Login Link", variant="primary")
            link = gr.Markdown()
            code = gr.Textbox(label="Step 2: Paste Google Auth Code (or full address)")
            cn = gr.Button("Connect & Sync Now", variant="primary")
            s1.click(ui_step1, [cs_up], [link, bdg])
            cn.click(ui_connect, [code], [link, bdg])
        with gr.Tab("1. Script & Story Writer"):
            t = gr.Textbox(label="Topic / Hook / Idea", lines=4)
            with gr.Row():
                f = gr.Dropdown(FORMATS, value=FORMATS[0], label="Creator Platform & Format")
                l = gr.Dropdown(LANGS, value=LANGS[0], label="Language & Vibe")
                to = gr.Dropdown(TONES, value=TONES[0], label="Tone / Emotion")
            ln = gr.Radio(list(LENGTHS), value="Medium (600-1000 words)", label="Length Mode")
            dr = gr.Textbox(label="Custom Director Instructions (characters, brand, twists, slang, rules)", lines=3)
            with gr.Row():
                b = gr.Button("Write Script", variant="primary")
                bx = gr.Button("➕ Continue / Extend Script (+1000 Words)")
            o = gr.Textbox(label="Script", lines=24); fo = gr.File(label="Download .txt"); st = gr.Textbox(label="Status")
            cur = gr.State(None)
            b.click(write_script, [t, f, l, to, ln, dr], [o, fo, st, cur])
            bx.click(extend_script, [o, f, l, to, dr, cur], [o, fo, st, cur])
        with gr.Tab("2. Human Text-to-Voice"):
            tx = gr.Textbox(label="Script Text", lines=10)
            with gr.Row():
                vv = gr.Dropdown(list(VOICES), value=list(VOICES)[0], label="Language & Voice")
                sy = gr.Dropdown(list(STYLES), value="Natural Chat", label="Delivery Style")
            with gr.Row():
                sp = gr.Slider(0.70, 1.35, value=1.0, step=0.01, label="Speed (x)")
                pi = gr.Slider(-20, 20, value=0, step=1, label="Pitch (Hz)")
                pa = gr.Slider(0.0, 1.0, value=0.5, step=0.05, label="Paragraph Breath Pause (s)")
            nt = gr.Textbox(label="Custom Pronunciation Notes (e.g. 'Kaggle=Kaa-gel, AI=A I')")
            b2 = gr.Button("Generate Voice", variant="primary")
            a2 = gr.Audio(label="Voice (.wav)", type="filepath"); m2 = gr.File(label="Download .mp3"); s2 = gr.Textbox(label="Status")
            b2.click(tts, [tx, vv, sy, sp, pi, pa, nt], [a2, m2, s2])
        with gr.Tab("3. Music & Song Studio"):
            with gr.Tab("Mode A - Long Instrumental / BGM (up to 3 min)"):
                mp = gr.Textbox(label="Music Prompt", value="cinematic lo-fi piano with soft rain, emotional")
                md = gr.Slider(30, 180, value=60, step=5, label="Duration (seconds)")
                b3 = gr.Button("Generate Music", variant="primary")
                a3 = gr.Audio(type="filepath"); s3 = gr.Textbox(label="Status")
                b3.click(music_mode_a, [mp, md], [a3, s3])
            with gr.Tab("Mode B - Full Song with Human Vocals"):
                ly = gr.Textbox(label="Lyrics ([Verse], [Chorus])", lines=10,
                                value="[Verse]\nTere bina yeh raatein\nKitni lambi lagti hain\n[Chorus]\nLaut aa, laut aa")
                with gr.Row():
                    sg = gr.Dropdown(list(SINGERS), value="Female", label="Singer")
                    gn = gr.Dropdown(list(GENRES), value="Bollywood Acoustic", label="Genre")
                cp = gr.Textbox(label="Extra Music Prompt")
                sd = gr.Slider(15, 180, value=45, step=5, label="Duration (seconds)")
                b4 = gr.Button("Generate Song", variant="primary")
                a4 = gr.Audio(type="filepath"); s4 = gr.Textbox(label="Status")
                b4.click(music_mode_b, [ly, sg, gn, cp, sd], [a4, s4])
        with gr.Tab("4. Sound Effects"):
            en = gr.Dropdown(list(ENV), value="Horror Ambience", label="Preset")
            de = gr.Textbox(label="Extra / Custom SFX Description")
            du = gr.Slider(2, 15, value=6, step=1, label="Duration (seconds)")
            b5 = gr.Button("Generate SFX", variant="primary")
            a5 = gr.Audio(type="filepath"); s5 = gr.Textbox(label="Status")
            b5.click(sfx, [de, en, du], [a5, s5])
        with gr.Tab("5. Titles, Hooks, Captions & Hashtags"):
            tp = gr.Textbox(label="Video Topic or Script", lines=6)
            with gr.Row():
                pf = gr.Dropdown(PLATFORMS, value=PLATFORMS[0], label="Platform")
                lg = gr.Dropdown(LANGS, value=LANGS[0], label="Language")
            cs = gr.Textbox(label="Extra Instructions")
            b6 = gr.Button("Generate Pack", variant="primary")
            o6 = gr.Textbox(label="Result", lines=18); f6 = gr.File(); s6 = gr.Textbox(label="Status")
            b6.click(seo, [tp, pf, lg, cs], [o6, f6, s6])
        with gr.Tab("6. Vocal Splitter & ZIP"):
            up = gr.File(label="Upload song or video")
            b7 = gr.Button("Split Vocals & Music", variant="primary")
            v7 = gr.Audio(label="Vocals", type="filepath"); m7 = gr.Audio(label="Music only", type="filepath"); s7 = gr.Textbox(label="Status")
            b7.click(split, [up], [v7, m7, s7])
            gr.Markdown("---")
            bz = gr.Button("Download AI_Studio_Master_Hub.zip", variant="primary"); fz = gr.File()
            bz.click(make_zip, None, fz)
    return ui

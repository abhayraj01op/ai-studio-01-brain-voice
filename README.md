# 🧠🎙️ AI Studio 01 — Brain & Voice

## ▶️ [1-Click Launch on Kaggle](https://kaggle.com/kernels/welcome?src=https://github.com/abhayraj01op/ai-studio-01-brain-voice/blob/main/Studio_01.ipynb)

Kaggle settings: **Accelerator: GPU T4 x2**, **Internet: ON**. Run Cell 1, then Cell 2, and open the `gradio.live` link.

## Tools (zero paid APIs)
1. **Script Writer** — Hermes-3-Llama-3.1-8B (default) or Qwen2.5-7B-Instruct (fp16, `device_map="auto"`), hard language lock (Hindi, Hinglish, Marathi, Bengali…), Bold/Unfiltered mode, unlimited "Continue / Extend".
2. **Text-to-Voice** — 22 voices (edge-tts + Kokoro-82M), 12 speaking styles, speed, pitch, breath pauses. If a voice is not offered by Microsoft, it falls back to another voice of the same language.
3. **Music** — MusicGen-medium BGM up to 5 min (sliding window + crossfade) and Bark-small sung vocals layered over a MusicGen backing track (20 singer presets, 25+ genres).
4. **Sound Effects** — AudioLDM2.
5. **Titles & Hashtags** — titles, captions, pinned comment, 20 hashtags.
6. **Splitter & Export** — Demucs vocal/instrumental split + whole Master Hub as ZIP.

All output goes to `/kaggle/working/AI_Studio_Master_Hub/` (`00_reference_faces` … `07_final_exports`, plus `99_model_cache` used as `HF_HOME`).

## Google Drive Live Sync
1. In Google Cloud Console create an OAuth client (type **Desktop app**), download `client_secrets.json`.
2. Upload it in the "☁️ Google Drive Live Sync Setup" panel (or attach it as a Kaggle dataset).
3. Step 1 → sign in → paste the code → **Connect & Sync Now**. Every new file is then uploaded automatically.

> Note: Google has restricted the copy-paste ("out-of-band") sign-in flow for new OAuth clients. If the sign-in page refuses, use an OAuth client created as a Desktop app in a project in "Testing" mode with your account added as a test user.

## 1-Time Model Cache
Press **⬇️ Download All Models Once** to save every model into `99_model_cache`. When Drive is connected, the cache is uploaded to Drive in the background (about 60 GB, needs that much free Drive space and takes a long time). In later sessions, connecting Drive downloads it back so models are not fetched from HuggingFace again.

## Tips
- Use **🗑️ Clear VRAM** when switching between heavy tools. Only one large model is kept loaded at a time.
- Every button shows the full error details if something fails.

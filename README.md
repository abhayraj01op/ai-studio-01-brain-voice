# 🧠🎙️ AI Studio 01 — Brain & Voice

## ▶️ [1-Click Launch on Kaggle](https://kaggle.com/kernels/welcome?src=https://github.com/abhayraj01op/ai-studio-01-brain-voice/blob/main/Studio_01.ipynb)

Kaggle settings: **Accelerator: GPU T4 x2**, **Internet: ON**. Run Cell 1, then Cell 2, and open the `gradio.live` link.

## Tools (zero paid APIs)
1. **Script Writer** — Qwen2.5-7B-Instruct (fp16, `device_map="auto"`), humanized, with unlimited "Continue / Extend".
2. **Text-to-Voice** — edge-tts neural voices (Hindi, Urdu, Indian/US English) + Kokoro-82M; speed, pitch, breath pauses.
3. **Music** — MusicGen-medium long BGM up to 3 min (sliding window + crossfade) and Bark sung vocals.
4. **Sound Effects** — AudioLDM2.
5. **Titles & Hashtags** — titles, captions, pinned comment, 20 hashtags.
6. **Splitter & Export** — Demucs vocal/instrumental split + whole Master Hub as ZIP.

All output goes to `/kaggle/working/AI_Studio_Master_Hub/` (`00_reference_faces` … `07_final_exports`).

## Google Drive Live Sync
1. In Google Cloud Console create an OAuth client (type **Desktop app**), download `client_secrets.json`.
2. Upload it in the "☁️ Google Drive Live Sync Setup" panel (or attach it as a Kaggle dataset).
3. Step 1 → sign in → paste the code → **Connect & Sync Now**. Every new file is then uploaded automatically.

> Note: Google has restricted the copy-paste ("out-of-band") sign-in flow for new OAuth clients. If the sign-in page refuses, use an OAuth client created as a Desktop app in a project in "Testing" mode with your account added as a test user.

## Tips
- Use **🗑️ Clear VRAM** when switching between heavy tools. Only one large model is kept loaded at a time.
- Every button shows the full error details if something fails.

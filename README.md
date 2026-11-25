# Studio 01 - Brain, Voice & Audio (v3)

[![Open in Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/abhayraj01op/ai-creator-studio/blob/main/Studio_01_Brain_Voice_Audio.ipynb)

1-click link: https://kaggle.com/kernels/welcome?src=https://github.com/abhayraj01op/ai-creator-studio/blob/main/Studio_01_Brain_Voice_Audio.ipynb

## Steps
1. Upload `app_studio1.py` and `Studio_01_Brain_Voice_Audio.ipynb` to the GitHub repo `abhayraj01op/ai-creator-studio` (branch `main`).
2. Open the link above. In Kaggle settings choose **GPU T4 x2** and turn **Internet ON**.
3. Click **Run All**, then open the public Gradio link.

## Tools
1. Script & Story Writer - Hermes-3-Llama-3.1-8B (fallback Qwen2.5-14B), 4-bit, 10 creator formats, 5 languages, 6 tones, 3 length modes + "Continue / Extend (+1000 words)" for unlimited length, director box -> `01_scripts`
2. Human Text-to-Voice - edge-tts (Hindi, Hinglish, Indian English, Urdu, Bengali, Marathi, Tamil, US/UK) + Kokoro, 5 delivery styles, speed 0.70-1.35x, pitch -20..+20 Hz, breath pause 0-1 s, pronunciation notes -> `02_voices` (.wav + 320k .mp3)
3. Music & Song Studio - Mode A: MusicGen-medium up to 3 min (sliding window + crossfade). Mode B: Bark singing vocals mixed over MusicGen backing -> `05_music`
4. Sound Effects - AudioLDM2, 6 presets, 2-15 s -> `05_music`
5. Hooks, Captions & Hashtags - same writer model: 5 hooks/titles, 3 captions, pinned comment, 20 hashtags -> `06_captions_hashtags`
6. Vocal Splitter (Demucs) + `AI_Studio_Master_Hub.zip` download

Only one heavy model stays in GPU memory; it is cleared (`gc.collect(); torch.cuda.empty_cache()`) when you switch tools. The header also has a "Clear VRAM / Free GPU Memory" button.

## Notes
- edge-tts voices need Internet ON (free Microsoft voices, no key).
- Bark singing is experimental: results vary per run; regenerate if a line sounds off.
- First run of each tool downloads the model (writer model ~5-9 GB).

## Google Drive sync (optional, inside the app)
Create a free OAuth "Desktop app" client in Google Cloud Console and download it as `client_secrets.json`. In the app open **Google Drive Live Sync Setup**, upload the file (or attach it as a Kaggle dataset - it is found automatically), click **Step 1**, sign in, paste the returned address in **Step 2**, then **Connect & Sync Now**. The badge turns to LIVE CONNECTED. Without it, files are saved only in `/kaggle/working/AI_Studio_Master_Hub/`. Existing Drive folders are reused, never deleted.

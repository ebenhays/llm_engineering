---
title: Multilingual Text Summarizer
emoji: 🌍
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: 4.0.0
app_file: app.py
pinned: false
license: mit
---

# Multilingual Text Summarizer with Voice Output

Paste text or upload a `.txt` file and get a concise AI-powered summary in any of **12 supported languages**. Optionally have the summary read aloud.

## Features

- Automatic source language detection
- Summarization via Claude 3.5 Sonnet (OpenRouter)
- 12 target languages: English, Spanish, French, German, Italian, Portuguese, Chinese, Japanese, Korean, Arabic, Hindi, Russian
- Text-to-speech playback (gTTS)
- Max 500 words input / 1MB file upload

## Setup

Add your `OPENROUTER_API_KEY` as a [Space secret](https://huggingface.co/docs/hub/spaces-overview#managing-secrets).

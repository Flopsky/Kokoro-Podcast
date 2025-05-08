
# **Kokoro-Podcast**

Open-source project that transforms arbitrarily long text into a continuous, streaming podcast-like audio experience. Its key innovation lies in its **streaming architecture**: it doesn't wait to process the entire text before playback. Instead, it begins narrating the text as soon as the first portion is ready, while simultaneously generating and queuing the remainder of the content in real time.

### Core Features:

* 🎙️ **Text-to-Speech (TTS)**: Converts text into natural-sounding speech using open or proprietary TTS engines.
* 📡 **Streaming Playback**: Plays the beginning of the podcast immediately while continuing to process and stream the rest of the text in the background.
* 🧵 **Chunked Generation**: Breaks down long texts into manageable segments that are queued and played one after another, giving the effect of an uninterrupted podcast.
* 📜 **Input Flexibility**: Accepts various sources of text input—articles, books, transcriptions, or user input.
* 🚀 **Efficient Resource Use**: Ideal for long-form listening without overloading memory, since it doesn't process everything upfront.

### Use Cases:

* Listening to articles, blog posts, or long documents on the go.
* Creating personal podcasts from reading material.
* Accessibility tool for users who prefer audio content.

## Components

1.  **Flask Backend (`tts_streaming/api/tts.py`):**
    *   Loads the Kokoro TTS model (`KPipeline`).
    *   Provides an API endpoint (`/api/tts-stream`) using Server-Sent Events (SSE) to stream generated audio chunks **(used by the Next.js frontend)**.
    *   Handles text processing and voice selection.

2.  **Gradio Frontend (`app_gradio.py`):**
    *   **Quick Start Option:** A simple, self-contained Gradio interface that runs the TTS model directly.
    *   Ideal for quickly testing the core TTS functionality without needing a separate backend process.
    *   Currently configured for **non-streaming** output (waits for full audio generation before playback).

3.  **Next.js Frontend (`nextjs-frontend/`):**
    *   A more modern web UI built with Next.js, React, TypeScript, and Tailwind CSS.
    *   Connects to the Flask backend's `/api/tts-stream` endpoint.
    *   Uses the Web Audio API to receive and play **streamed** audio chunks, allowing playback to start before full generation is complete.
    *   Provides replay functionality once the stream is finished.
    *   Requires running the Flask backend separately.

## Prerequisites

*   **Python:** Version 3.10+ recommended.
*   **Node.js & npm:** Required *only* for the Next.js frontend (v18+ recommended).
*   **pip:** Python package installer.
*   **FFmpeg:** Required by Gradio and underlying audio libraries.

## Installation

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/Flopsky/kokoro-streamer.git
    cd kokoro-streamer
    ```

2.  **Install System Dependencies (FFmpeg):**
    *   Debian/Ubuntu: `sudo apt-get update && sudo apt-get install -y ffmpeg`
    *   macOS (Homebrew): `brew install ffmpeg`
    *   Windows: Download from [FFmpeg website](https://ffmpeg.org/download.html) and add to PATH.

3.  **Install Python Packages:**
    *(Recommended to use a virtual environment)*
    ```bash
    # python -m venv venv
    # source venv/bin/activate 
    pip install -r requirements.txt
    ```

4.  **Install Next.js Dependencies (Only if using Next.js frontend):**
    ```bash
    cd nextjs-frontend
    npm install
    cd ..
    ```

## Running the Application

**Option 1: Gradio UI (Quickest Start, Non-Streaming)**

1.  **Run the Gradio script:**
    *(Ensure you are in the project root directory)*
    ```bash
    python app_gradio.py
    ```
    *(This runs the Gradio app, likely on port 53940 or the next available. It includes the TTS model directly.)*

2.  **Access:** Open your browser to the URL shown in the terminal.

---

**Option 2: Next.js Frontend (Requires Separate Backend, Streaming Playback)**

1.  **Start the Flask Backend:**
    *(Ensure you are in the project root directory)*
    ```bash
    python tts_streaming/api/tts.py
    ```
    *(Keep this terminal running. The backend runs on port 57103)*

2.  **Start the Next.js Frontend:**
    *(Open a second terminal in the project root directory)*
    ```bash
    cd nextjs-frontend
    npm run dev -- -p 53940 
    ```
    *(This runs the frontend on port 53940)*

3.  **Access:** Open your browser to `http://localhost:53940`.

**First Run Note:** The first time you run either application, the necessary Kokoro TTS models and voice files will be downloaded from Hugging Face Hub. This might take some time.

## Usage (Next.js Frontend)

1.  Enter text.
2.  Select a voice.
3.  Click "Forge Voice".
4.  Audio playback will start automatically as chunks are received.
5.  Once finished, a standard audio player appears allowing replay.

## Notes

*   **Sample Rate:** Assumed to be 24000 Hz.
*   **Streaming:** Only the Next.js frontend currently implements true audio streaming playback.
*   **Phonemizer Warnings:** Console warnings like `words count mismatch` can often be ignored if audio sounds correct.

## License

Kokoro library uses Apache 2.0. Please respect its terms and those of its dependencies.

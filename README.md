# Kokoro TTS Streamer

This project provides web interfaces (Gradio and Next.js) for text-to-speech (TTS) synthesis using the Kokoro TTS model (`hexgrad/Kokoro-82M`).

## Components

1.  **Flask Backend (`openhands/api/tts.py`):**
    *   Loads the Kokoro TTS model (`KPipeline`).
    *   Provides an API endpoint (`/api/tts-stream`) using Server-Sent Events (SSE) to stream generated audio chunks.
    *   Handles text processing and voice selection.

2.  **Gradio Frontend (`app_gradio.py`):**
    *   A simple, self-contained Gradio interface.
    *   Connects directly to the Kokoro TTS pipeline within the same script.
    *   Currently configured for **non-streaming** output (waits for full audio generation).

3.  **Next.js Frontend (`nextjs-frontend/`):**
    *   A modern web UI built with Next.js, React, TypeScript, and Tailwind CSS.
    *   Connects to the Flask backend's `/api/tts-stream` endpoint.
    *   Uses the Web Audio API to receive and play **streamed** audio chunks, allowing playback to start before full generation is complete.
    *   Provides replay functionality once the stream is finished.

## Prerequisites

*   **Python:** Version 3.10+ recommended.
*   **Node.js & npm:** Required for the Next.js frontend (v18+ recommended).
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
    pip install -r requirements.txt # Or install manually:
    # pip install Flask Flask-Cors Flask-SSE gradio torch numpy soundfile kokoro==0.9.4 huggingface_hub
    ```
    *(Note: A `requirements.txt` doesn't exist yet, manual install is needed for now. Torch installation might need customization based on your system/GPU - see PyTorch website.)*

4.  **Install Next.js Dependencies:**
    ```bash
    cd nextjs-frontend
    npm install
    cd ..
    ```

## Running the Application

You can run either the Gradio UI or the Next.js UI.

**Option 1: Running the Next.js Frontend (with Streaming)**

1.  **Start the Flask Backend:**
    *(Ensure you are in the project root directory)*
    ```bash
    python openhands/api/tts.py
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

**Option 2: Running the Gradio Frontend (Non-Streaming)**

1.  **Run the Gradio script:**
    *(Ensure you are in the project root directory)*
    ```bash
    python app_gradio.py
    ```
    *(This runs the Gradio app, likely on port 53940 or the next available)*

2.  **Access:** Open your browser to the URL shown in the terminal.

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

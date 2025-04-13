import os
import io
import time
import logging
import base64
from flask import Flask, request, jsonify, Response, stream_with_context
from flask_cors import CORS
import torch
import numpy as np
import soundfile as sf
from kokoro.pipeline import KPipeline

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO)

# --- Global TTS instance ---
tts_instance = None
kokoro_available = False
SAMPLE_RATE = 24000 # Assuming 24kHz

try:
    start_time = time.time()
    logging.info("Initializing Kokoro KPipeline...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tts_instance = KPipeline(lang_code='a', device=device, repo_id='hexgrad/Kokoro-82M')
    kokoro_available = True
    end_time = time.time()
    logging.info(f"Kokoro KPipeline initialized successfully on device '{device}' in {end_time - start_time:.2f} seconds.")
except Exception as e:
    logging.error(f"Failed to initialize Kokoro KPipeline: {e}", exc_info=True)

# --- Flask App ---
app = Flask(__name__)
CORS(app) # Enable CORS

# --- Helper Function to convert audio chunk to WAV bytes ---
def samples_chunk_to_wav_bytes(samples: np.ndarray, sample_rate: int) -> bytes:
    if samples is None or samples.size == 0:
        return b''
    buffer = io.BytesIO()
    if samples.dtype != np.float32:
        samples = samples.astype(np.float32)
    try:
        sf.write(buffer, samples, sample_rate, format='WAV', subtype='PCM_16')
        buffer.seek(0)
        return buffer.read()
    except Exception as e:
        logging.error(f"Error writing WAV chunk data: {e}", exc_info=True)
        return b''

# --- SSE Streaming Endpoint (Now accepts GET) ---
@app.route('/api/tts-stream', methods=['GET']) # Changed to GET
def handle_tts_stream_request():
    if not kokoro_available or tts_instance is None:
        # Return error in SSE format for consistency
        return Response("event: stream_error\ndata: TTS service not available\n\n", mimetype='text/event-stream', status=503)

    # Get parameters from query string
    text = request.args.get('text')
    voice = request.args.get('voice')

    if not text or not voice:
        return Response("event: stream_error\ndata: Missing 'text' or 'voice' query parameter\n\n", mimetype='text/event-stream', status=400)
    if not text.strip():
         return Response("event: stream_error\ndata: Text cannot be empty\n\n", mimetype='text/event-stream', status=400)

    def generate_audio_stream():
        logging.info(f"Starting stream for voice='{voice}', text='{text[:50]}...'")
        try:
            synthesis_generator = tts_instance(text=text, voice=voice)
            chunk_index = 0
            for item in synthesis_generator:
                chunk_index += 1
                logging.info(f"Processing chunk {chunk_index}")
                if hasattr(item, 'output') and hasattr(item.output, 'audio') and item.output.audio is not None:
                    audio_tensor = item.output.audio
                    if hasattr(audio_tensor, 'detach'):
                        chunk_samples = audio_tensor.detach().cpu().numpy()
                    else:
                        chunk_samples = np.array(audio_tensor)
                    if chunk_samples.ndim > 1:
                        chunk_samples = chunk_samples.squeeze()
                    
                    if chunk_samples.size > 0:
                        wav_chunk_bytes = samples_chunk_to_wav_bytes(chunk_samples, SAMPLE_RATE)
                        if wav_chunk_bytes:
                            base64_chunk = base64.b64encode(wav_chunk_bytes).decode('utf-8')
                            sse_message = f"event: audio_chunk\ndata: {base64_chunk}\n\n"
                            yield sse_message
                            logging.info(f"Sent chunk {chunk_index} ({len(wav_chunk_bytes)} bytes)")
                        else:
                            logging.warning(f"Skipped empty WAV chunk {chunk_index}.")
                    else:
                         logging.warning(f"Skipped empty numpy chunk {chunk_index}.")
                else:
                    logging.warning(f"Generator yielded item without audio in chunk {chunk_index}")
            
            yield "event: stream_end\ndata: finished\n\n"
            logging.info("Stream finished event sent.")

        except Exception as e:
            logging.error(f"Error during TTS stream generation: {e}", exc_info=True)
            error_message = base64.b64encode(str(e).encode('utf-8')).decode('utf-8')
            yield f"event: stream_error\ndata: {error_message}\n\n"

    return Response(stream_with_context(generate_audio_stream()), mimetype='text/event-stream')

# --- Main Execution ---
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=57103, debug=False, threaded=True)

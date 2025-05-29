import gradio as gr
import numpy as np
import torch
import logging
import time
import io
import soundfile as sf
from kokoro.pipeline import KPipeline
from script_generator.script_generator import generate_script

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO)

# --- Global TTS instance ---
tts_instance = None
kokoro_available = False
SAMPLE_RATE = 24000 # Assuming 24kHz, adjust if needed

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

# --- Voice Options ---
voice_options = [
    'af_alloy', 'af_aoede', 'af_bella', 'af_heart', 'af_jessica', 
    'af_kore', 'af_nicole', 'af_nova', 'af_river', 'af_sarah', 'af_sky',
    'am_echo', 'bf_alice', 'bm_daniel', 'jf_gongitsune', 'zm_yunxi' 
]

def prepare_audio_for_gradio(samples: np.ndarray, sample_rate: int):
    """Prepare numpy audio samples for Gradio streaming"""
    if samples is None or samples.size == 0:
        return None
    
    # Ensure samples are float32
    if samples.dtype != np.float32:
        samples = samples.astype(np.float32)
    
    # Ensure samples are in the right shape (1D)
    if samples.ndim > 1:
        samples = samples.squeeze()
    
    # Normalize to [-1, 1] range if needed
    max_val = np.max(np.abs(samples))
    if max_val > 1.0:
        samples = samples / max_val
    
    # Return tuple format that Gradio expects
    return (sample_rate, samples)

# --- Gradio Processing Function (Streaming) ---
def generate_speech_streaming(text, voice):
    if not kokoro_available or tts_instance is None:
        yield None, "Error: TTS Service initialization failed. Check logs."
        return 
    if not text or not text.strip():
        yield None, "Error: Please enter some text."
        return 
    if not voice:
        yield None, "Error: Please select a voice."
        return

    logging.info(f"Received streaming request: voice='{voice}', text='{text[:50]}...'")
    
    # Clear previous output and show processing status
    yield None, "Processing text..."
    
    try:
        start_time = time.time()
        script = generate_script(text)
        synthesis_generator = tts_instance(text=script, voice=voice)
        
        chunk_index = 0
        all_audio_chunks = [] # Collect all chunks here
        
        for item in synthesis_generator:
            chunk_index += 1
            status_update = f"Generating audio chunk {chunk_index}..."
            logging.info(status_update)

            if hasattr(item, 'output') and hasattr(item.output, 'audio') and item.output.audio is not None:
                audio_tensor = item.output.audio
                if hasattr(audio_tensor, 'detach'):
                    chunk_samples = audio_tensor.detach().cpu().numpy()
                else:
                    chunk_samples = np.array(audio_tensor)
                
                if chunk_samples.ndim > 1:
                    chunk_samples = chunk_samples.squeeze()
                
                if chunk_samples.size > 0:
                    all_audio_chunks.append(chunk_samples)
                    logging.info(f"Streaming audio chunk {chunk_index} with shape: {chunk_samples.shape}")
                    
                    # Prepare audio for Gradio streaming
                    audio_data = prepare_audio_for_gradio(chunk_samples, SAMPLE_RATE)
                    if audio_data is not None:
                        yield audio_data, status_update
                    else:
                        yield None, f"Error processing chunk {chunk_index}"
                else:
                    logging.warning(f"Skipping empty audio chunk {chunk_index}.")
                    yield None, status_update
            else:
                logging.warning(f"Generator yielded item without expected audio data in chunk {chunk_index}: {item}")
                yield None, status_update
        
        end_time = time.time()
        final_status = f"Finished generating {chunk_index} chunks in {end_time - start_time:.2f} seconds."
        logging.info(final_status)
        
        # Yield final status (audio streaming continues with last chunk)
        yield None, final_status
        
    except Exception as e:
        error_message = f"Error during streaming TTS generation: {e}"
        logging.error(error_message, exc_info=True)
        yield None, f"Error: {e}"

# --- Non-streaming fallback function ---
def generate_speech_complete(text, voice):
    """Generate complete audio (fallback if streaming fails)"""
    if not kokoro_available or tts_instance is None:
        return None, "Error: TTS Service initialization failed. Check logs."
    if not text or not text.strip():
        return None, "Error: Please enter some text."
    if not voice:
        return None, "Error: Please select a voice."

    logging.info(f"Received complete generation request: voice='{voice}', text='{text[:50]}...'")
    
    try:
        start_time = time.time()
        script = generate_script(text)
        synthesis_generator = tts_instance(text=script, voice=voice)
        
        all_audio_chunks = []
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
                    all_audio_chunks.append(chunk_samples)
        
        if all_audio_chunks:
            final_audio = np.concatenate(all_audio_chunks)
            audio_data = prepare_audio_for_gradio(final_audio, SAMPLE_RATE)
            
            end_time = time.time()
            final_status = f"Generated complete audio in {end_time - start_time:.2f} seconds."
            return audio_data, final_status
        else:
            return None, "No audio generated."
            
    except Exception as e:
        error_message = f"Error during TTS generation: {e}"
        logging.error(error_message, exc_info=True)
        return None, f"Error: {e}"

# --- Gradio Interface Definition ---
with gr.Blocks(theme=gr.themes.Soft(primary_hue="blue", secondary_hue="sky"), css=".gradio-container { max-width: 800px !important; margin: auto; }") as demo:
    gr.Markdown("## ✨ Kokoro TTS Streamer ✨")
    gr.Markdown("Enter your text, choose a voice, and hear the magic happen in real-time!")
    
    with gr.Row():
        streaming_mode = gr.Checkbox(label="Enable Streaming", value=True)
    
    with gr.Row():
        with gr.Column(scale=2):
            text_input = gr.Textbox(label="Input Text", lines=7, placeholder="Enter long text here...")
            voice_dropdown = gr.Dropdown(label="Select Voice", choices=voice_options, value=voice_options[0])
            submit_button = gr.Button("🎤 Generate Speech", variant="primary")
        with gr.Column(scale=1):
            status_indicator = gr.Markdown("Status: Ready") 
            audio_output = gr.Audio(label="Output Audio", streaming=True, autoplay=True) 

    def process_request(text, voice, use_streaming):
        if use_streaming:
            # For streaming, we need to yield from the generator
            yield from generate_speech_streaming(text, voice)
        else:
            # For non-streaming, return the result directly
            return generate_speech_complete(text, voice)

    submit_button.click(
        fn=process_request,
        inputs=[text_input, voice_dropdown, streaming_mode],
        outputs=[audio_output, status_indicator]
    )

# --- Launch the App ---
if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=True)

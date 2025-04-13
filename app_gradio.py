import gradio as gr
import numpy as np
import torch
import logging
import time
import io
import soundfile as sf
from kokoro.pipeline import KPipeline

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

# --- Gradio Processing Function (Streaming) ---
def generate_speech_streaming(text, voice):
    # Define component references here to yield updates
    audio_output_comp = audio_output # Reference the gr.Audio component
    status_indicator_comp = status_indicator # Reference the gr.Markdown component
    
    if not kokoro_available or tts_instance is None:
        yield {status_indicator_comp: gr.Markdown("Error: TTS Service initialization failed. Check logs."), audio_output_comp: None}
        return 
    if not text or not text.strip():
        yield {status_indicator_comp: gr.Markdown("Error: Please enter some text."), audio_output_comp: None}
        return 
    if not voice:
        yield {status_indicator_comp: gr.Markdown("Error: Please select a voice."), audio_output_comp: None}
        return

    logging.info(f"Received streaming request: voice='{voice}', text='{text[:50]}...'")
    
    # Clear previous output and show processing status
    yield {audio_output_comp: None, status_indicator_comp: gr.Markdown("Processing text...")}
    
    try:
        start_time = time.time()
        synthesis_generator = tts_instance(text=text, voice=voice)
        
        chunk_index = 0
        all_audio_chunks = [] # Collect all chunks here
        
        for item in synthesis_generator:
            chunk_index += 1
            status_update = f"Generating audio chunk {chunk_index}..."
            # Yield status update first
            yield {status_indicator_comp: gr.Markdown(status_update)}
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
                    all_audio_chunks.append(chunk_samples) # Collect the chunk
                    logging.info(f"Yielding audio chunk {chunk_index} with shape: {chunk_samples.shape}")
                    # Yield the chunk for streaming playback
                    yield {audio_output_comp: (SAMPLE_RATE, chunk_samples), status_indicator_comp: gr.Markdown(status_update)}
                else:
                    logging.warning(f"Skipping empty audio chunk {chunk_index}.")
                    # Still yield status update if chunk is empty
                    # yield {status_indicator_comp: gr.Markdown(status_update)} 
            else:
                logging.warning(f"Generator yielded item without expected audio data in chunk {chunk_index}: {item}")
                # Still yield status update if no audio data
                # yield {status_indicator_comp: gr.Markdown(status_update)} 
        
        end_time = time.time()
        final_status = f"Finished generating {chunk_index} chunks in {end_time - start_time:.2f} seconds."
        logging.info(final_status)
        
        # After loop, yield the final status and None for the audio to finalize the player
        yield {status_indicator_comp: gr.Markdown(final_status), audio_output_comp: None}
        
        
    except Exception as e:
        error_message = f"Error during streaming TTS generation: {e}"
        logging.error(error_message, exc_info=True)
        yield {status_indicator_comp: gr.Markdown(f"Error: {e}")}

# --- Gradio Interface Definition ---
with gr.Blocks(theme=gr.themes.Soft(primary_hue="blue", secondary_hue="sky"), css=".gradio-container { max-width: 800px !important; margin: auto; }") as demo:
    gr.Markdown("## ✨ Kokoro TTS Streamer ✨")
    gr.Markdown("Enter your text, choose a voice, and hear the magic happen in real-time!")
    with gr.Row():
        with gr.Column(scale=2):
            text_input = gr.Textbox(label="Input Text", lines=7, placeholder="Enter long text here...")
            voice_dropdown = gr.Dropdown(label="Select Voice", choices=voice_options, value=voice_options[0])
            submit_button = gr.Button("🎤 Generate Speech", variant="primary")
        with gr.Column(scale=1):
            # Define components with variable names matching those used in the streaming function
            status_indicator = gr.Markdown("Status: Ready") 
            # Re-enable streaming
            audio_output = gr.Audio(label="Output Audio", streaming=True, autoplay=True) 

    submit_button.click(
        fn=generate_speech_streaming, # Use the STREAMING function
        inputs=[text_input, voice_dropdown],
        outputs=[audio_output, status_indicator] # Target both outputs
    )

# --- Launch the App ---
if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=53940)

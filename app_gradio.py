import gradio as gr
import numpy as np
import torch
import logging
import time
import io
import soundfile as sf
import platform
import subprocess
import sys
from kokoro.pipeline import KPipeline

# Try to import script generator, make it optional
try:
    from script_generator.script_generator import generate_script
    SCRIPT_GENERATOR_AVAILABLE = True
except Exception as e:
    logging.warning(f"Script generator not available: {e}")
    SCRIPT_GENERATOR_AVAILABLE = False
    def generate_script(text):
        return text  # Return original text as fallback

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO)

# --- Apple Silicon Detection ---
def is_apple_silicon():
    """Detect if running on Apple Silicon"""
    try:
        return platform.system() == "Darwin" and platform.machine() == "arm64"
    except:
        return False

# --- MLX Support ---
def install_mlx_audio():
    """Install mlx-audio package for Apple Silicon"""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-U", "mlx-audio"])
        return True
    except subprocess.CalledProcessError:
        logging.error("Failed to install mlx-audio")
        return False

MLX_AVAILABLE = False
if is_apple_silicon():
    try:
        import mlx_audio
        MLX_AVAILABLE = True
        logging.info("MLX Audio detected - using optimized Apple Silicon version")
    except ImportError:
        logging.info("MLX Audio not found, attempting to install...")
        if install_mlx_audio():
            try:
                import mlx_audio
                MLX_AVAILABLE = True
                logging.info("MLX Audio installed successfully")
            except ImportError:
                logging.warning("MLX Audio installation failed, falling back to standard Kokoro")
                MLX_AVAILABLE = False
        else:
            MLX_AVAILABLE = False

# --- TTS Pipeline Setup ---
tts_instance = None
kokoro_available = False
SAMPLE_RATE = 24000

if MLX_AVAILABLE and is_apple_silicon():
    # Use MLX version for Apple Silicon
    try:
        start_time = time.time()
        logging.info("Initializing MLX Kokoro Pipeline for Apple Silicon...")
        from mlx_audio.tts.generate import generate_audio
        
        # Test MLX initialization with a simple call
        test_file = "test_mlx_init.wav"
        generate_audio(
            text="test", 
            model_path="mlx-community/Kokoro-82M-bf16",
            voice="af_heart",
            file_prefix=test_file.replace('.wav', ''),
            verbose=False,
            play=False
        )
        # Clean up test file
        import os
        if os.path.exists(test_file):
            os.remove(test_file)
            
        kokoro_available = True
        tts_type = "MLX"
        
        end_time = time.time()
        logging.info(f"MLX Kokoro Pipeline initialized successfully in {end_time - start_time:.2f} seconds.")
    except Exception as e:
        logging.error(f"Failed to initialize MLX Kokoro Pipeline: {e}", exc_info=True)
        MLX_AVAILABLE = False

if not MLX_AVAILABLE:
    # Fallback to standard Kokoro
    try:
        start_time = time.time()
        logging.info("Initializing standard Kokoro KPipeline...")
        from kokoro.pipeline import KPipeline
        device = "cuda" if torch.cuda.is_available() else "cpu"
        tts_instance = KPipeline(lang_code='a', device=device, repo_id='hexgrad/Kokoro-82M')
        kokoro_available = True
        tts_type = "Standard"
        end_time = time.time()
        logging.info(f"Standard Kokoro KPipeline initialized successfully on device '{device}' in {end_time - start_time:.2f} seconds.")
    except Exception as e:
        logging.error(f"Failed to initialize Kokoro KPipeline: {e}", exc_info=True)

# --- MLX Audio Generation (Streaming) ---
def generate_audio_mlx_streaming(text, voice):
    """Generate audio using MLX Kokoro with streaming support"""
    try:
        import tempfile
        import os
        import glob
        import time
        from mlx_audio.tts.generate import generate_audio
        
        # MLX Kokoro voice mapping
        mlx_voice_map = {
            'af_alloy': 'af_alloy', 'af_aoede': 'af_aoede', 'af_bella': 'af_bella',
            'af_heart': 'af_heart', 'af_jessica': 'af_jessica', 'af_kore': 'af_kore',
            'af_nicole': 'af_nicole', 'af_nova': 'af_nova', 'af_river': 'af_river',
            'af_sarah': 'af_sarah', 'af_sky': 'af_sky', 'am_echo': 'am_echo',
            'bf_alice': 'bf_alice', 'bm_daniel': 'bm_daniel', 
            'jf_gongitsune': 'jf_gongitsune', 'zm_yunxi': 'zm_yunxi'
        }
        
        mlx_voice = mlx_voice_map.get(voice, 'af_heart')
        
        # Create temporary directory for output files
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_file_path = os.path.join(temp_dir, "mlx_stream")
            
            # Start MLX generation with streaming enabled in a separate thread
            import threading
            generation_complete = threading.Event()
            
            def generate_in_background():
                try:
                    generate_audio(
                        text=text,
                        model_path="mlx-community/Kokoro-82M-bf16",
                        voice=mlx_voice,
                        file_prefix=temp_file_path,
                        verbose=False,
                        play=False,
                        audio_format='wav',
                        stream=True,
                        streaming_interval=1.0  # Generate chunks every 1 second
                    )
                finally:
                    generation_complete.set()
            
            # Start background generation
            generation_thread = threading.Thread(target=generate_in_background)
            generation_thread.start()
            
            # Stream audio chunks as they become available
            chunk_index = 0
            processed_files = set()
            
            while not generation_complete.is_set() or len(processed_files) < 50:  # Safety limit
                # Look for new audio files
                pattern = f"{temp_file_path}_*.wav"
                current_files = set(glob.glob(pattern))
                new_files = current_files - processed_files
                
                if new_files:
                    # Sort files to process them in order
                    new_files_sorted = sorted(new_files, key=lambda x: int(x.split('_')[-1].split('.')[0]))
                    
                    for file_path in new_files_sorted:
                        try:
                            # Load and yield the audio chunk
                            audio_data, file_sample_rate = sf.read(file_path)
                            
                            # Log the actual sample rate for debugging
                            logging.info(f"Loaded audio chunk with sample rate: {file_sample_rate}Hz, shape: {audio_data.shape}")
                            
                            # Convert to float32 and ensure correct shape
                            if audio_data.dtype != np.float32:
                                audio_data = audio_data.astype(np.float32)
                            
                            if audio_data.ndim > 1:
                                audio_data = audio_data.squeeze()
                            
                            if audio_data.size > 0:
                                chunk_index += 1
                                processed_files.add(file_path)
                                
                                # Prepare audio for Gradio with correct sample rate
                                audio_output = prepare_audio_for_gradio(audio_data, file_sample_rate)
                                yield audio_output, f"Streaming MLX chunk {chunk_index}..."
                                logging.info(f"Streamed MLX audio chunk {chunk_index} from {os.path.basename(file_path)}")
                            
                        except Exception as e:
                            logging.warning(f"Error processing MLX chunk {file_path}: {e}")
                
                if not generation_complete.is_set():
                    time.sleep(0.5)  # Check for new files every 0.5 seconds
                elif not new_files:
                    break  # No more files expected
            
            # Wait for background thread to complete
            generation_thread.join(timeout=10)
            
            # Final status
            if chunk_index > 0:
                yield None, f"MLX streaming complete - {chunk_index} chunks generated"
            else:
                yield None, "MLX generation failed - no audio chunks created"
                
    except Exception as e:
        logging.error(f"MLX streaming generation failed: {e}")
        yield None, f"MLX streaming error: {e}"

# --- MLX Audio Generation (Non-streaming fallback) ---
def generate_audio_mlx(text, voice):
    """Generate complete audio using MLX Kokoro (non-streaming fallback)"""
    try:
        import tempfile
        import os
        from mlx_audio.tts.generate import generate_audio
        
        mlx_voice_map = {
            'af_alloy': 'af_alloy', 'af_aoede': 'af_aoede', 'af_bella': 'af_bella',
            'af_heart': 'af_heart', 'af_jessica': 'af_jessica', 'af_kore': 'af_kore',
            'af_nicole': 'af_nicole', 'af_nova': 'af_nova', 'af_river': 'af_river',
            'af_sarah': 'af_sarah', 'af_sky': 'af_sky', 'am_echo': 'am_echo',
            'bf_alice': 'bf_alice', 'bm_daniel': 'bm_daniel', 
            'jf_gongitsune': 'jf_gongitsune', 'zm_yunxi': 'zm_yunxi'
        }
        
        mlx_voice = mlx_voice_map.get(voice, 'af_heart')
        
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
            temp_file_path = tmp_file.name.replace('.wav', '')
            
        try:
            # Generate complete audio with MLX
            generate_audio(
                text=text,
                model_path="mlx-community/Kokoro-82M-bf16",
                voice=mlx_voice,
                file_prefix=temp_file_path,
                verbose=False,
                play=False,
                audio_format='wav',
                join_audio=True  # Join all chunks into one file
            )
            
            # Find the output file (could be _000.wav or just .wav)
            possible_files = [
                temp_file_path + '_000.wav',
                temp_file_path + '.wav'
            ]
            
            output_file = None
            for file_path in possible_files:
                if os.path.exists(file_path):
                    output_file = file_path
                    break
            
            if output_file and os.path.exists(output_file):
                audio_data, sample_rate = sf.read(output_file)
                os.remove(output_file)
                
                logging.info(f"Loaded complete MLX audio with sample rate: {sample_rate}Hz, shape: {audio_data.shape}")
                
                if audio_data.dtype != np.float32:
                    audio_data = audio_data.astype(np.float32)
                
                if audio_data.ndim > 1:
                    audio_data = audio_data.squeeze()
                    
                return audio_data, sample_rate  # Return both audio and sample rate
            else:
                logging.error(f"MLX output file not found. Checked: {possible_files}")
                return None, None
                
        finally:
            # Clean up any remaining temp files
            import glob
            cleanup_pattern = temp_file_path + "*"
            for cleanup_file in glob.glob(cleanup_pattern):
                if os.path.exists(cleanup_file):
                    os.remove(cleanup_file)
        
    except Exception as e:
        logging.error(f"MLX audio generation failed: {e}")
        return None, None

# --- Voice Options ---
voice_options = [
    'af_alloy', 'af_aoede', 'af_bella', 'af_heart', 'af_jessica', 
    'af_kore', 'af_nicole', 'af_nova', 'af_river', 'af_sarah', 'af_sky',
    'am_echo', 'bf_alice', 'bm_daniel', 'jf_gongitsune', 'zm_yunxi' 
]

def prepare_audio_for_gradio(samples: np.ndarray, sample_rate: int):
    """Prepare numpy audio samples for Gradio streaming with proper format for playback and download"""
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
    
    # Gradio expects sample rate to be consistent - use the actual MLX/Kokoro sample rate
    # MLX Kokoro typically outputs at 24kHz, but let's make sure it's compatible
    target_sample_rate = 24000  # Kokoro's native sample rate
    
    # Return tuple format that Gradio expects: (sample_rate, audio_array)
    return (target_sample_rate, samples)

# --- Gradio Processing Function (Streaming) ---
def generate_speech_streaming(text, voice):
    if not kokoro_available:
        yield None, "Error: TTS Service initialization failed. Check logs."
        return 
    if not text or not text.strip():
        yield None, "Error: Please enter some text."
        return 
    if not voice:
        yield None, "Error: Please select a voice."
        return

    logging.info(f"Received streaming request: voice='{voice}', text='{text[:50]}...' using {tts_type} pipeline")
    
    # Clear previous output and show processing status
    yield None, "Processing text..."
    
    try:
        start_time = time.time()
        script = generate_script(text)
        
        if MLX_AVAILABLE and tts_type == "MLX":
            # Use MLX for Apple Silicon with streaming
            logging.info("Generating streaming audio with MLX Kokoro...")
            for audio_chunk, status_message in generate_audio_mlx_streaming(script, voice):
                if audio_chunk is not None and audio_chunk.size > 0:
                    audio_output = prepare_audio_for_gradio(audio_chunk, SAMPLE_RATE)
                    yield audio_output, status_message
                else:
                    yield None, status_message
            
            end_time = time.time()
            final_status = f"MLX streaming complete in {end_time - start_time:.2f} seconds."
            yield None, final_status
        else:
            # Use standard Kokoro with streaming
            synthesis_generator = tts_instance(text=script, voice=voice)
            
            chunk_index = 0
            all_audio_chunks = []
            
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
                        
                        # Prepare audio for Gradio streaming with correct sample rate
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
    if not kokoro_available:
        return None, "Error: TTS Service initialization failed. Check logs."
    if not text or not text.strip():
        return None, "Error: Please enter some text."
    if not voice:
        return None, "Error: Please select a voice."

    logging.info(f"Received complete generation request: voice='{voice}', text='{text[:50]}...' using {tts_type} pipeline")
    
    try:
        start_time = time.time()
        script = generate_script(text)
        
        if MLX_AVAILABLE and tts_type == "MLX":
            # Use MLX for Apple Silicon
            logging.info("Generating complete audio with MLX Kokoro...")
            audio_data, sample_rate = generate_audio_mlx(script, voice)
            
            if audio_data is not None and audio_data.size > 0:
                audio_output = prepare_audio_for_gradio(audio_data, sample_rate)
                end_time = time.time()
                final_status = f"Generated complete audio with MLX in {end_time - start_time:.2f} seconds."
                return audio_output, final_status
            else:
                return None, "Error: MLX audio generation failed."
        else:
            # Use standard Kokoro
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
                logging.info(f"Complete audio generated with shape: {final_audio.shape}, sample rate: {SAMPLE_RATE}Hz")
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
    
    # Show pipeline information
    if kokoro_available:
        if tts_type == "MLX":
            pipeline_info = "🚀 **Apple Silicon Optimized** - Using MLX Kokoro for enhanced performance"
        else:
            pipeline_info = f"⚡ **Standard Pipeline** - Using Kokoro on {device if 'device' in locals() else 'CPU'}"
    else:
        pipeline_info = "❌ **TTS Unavailable** - Pipeline initialization failed"
    
    gr.Markdown(pipeline_info)
    
    if SCRIPT_GENERATOR_AVAILABLE:
        gr.Markdown("Enter your text, choose a voice, and hear the magic happen in real-time! ✨ Script generation powered by Gemini AI.")
    else:
        gr.Markdown("⚠️ **Demo Mode**: Script generation disabled (Gemini API not configured). Text will be used directly for TTS.")
    
    with gr.Row():
        streaming_mode = gr.Checkbox(label="Enable Streaming", value=True)
    
    with gr.Row():
        with gr.Column(scale=2):
            text_input = gr.Textbox(label="Input Text", lines=7, placeholder="Enter long text here...")
            voice_dropdown = gr.Dropdown(label="Select Voice", choices=voice_options, value=voice_options[0])
            submit_button = gr.Button("🎤 Generate Speech", variant="primary")
        with gr.Column(scale=1):
            status_indicator = gr.Markdown("Status: Ready") 
            audio_output = gr.Audio(
                label="Output Audio", 
                streaming=True, 
                autoplay=True,
                show_download_button=True,
                show_share_button=True,
                type="numpy",
                format="wav"
            )

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
    import os
    port = int(os.environ.get("GRADIO_SERVER_PORT", 7860))
    demo.launch(server_name="0.0.0.0", server_port=port, share=True)

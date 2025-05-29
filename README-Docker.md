# Kokoro TTS Streamer - Docker Setup

This guide shows how to run the Kokoro TTS Streamer using Docker with optimizations for Apple Silicon devices.

## Prerequisites

- Docker installed on your system
- Docker Compose (optional, but recommended)

## Apple Silicon Optimization

This project includes special optimizations for Apple Silicon (M1/M2/M3) devices using [MLX Kokoro](https://huggingface.co/mlx-community/Kokoro-82M-bf16). MLX provides significant performance improvements on Apple Silicon hardware.

### Automatic Detection
The application automatically detects Apple Silicon and uses the optimized MLX pipeline when available.

### Manual MLX Installation
If you're running natively on macOS with Apple Silicon:
```bash
pip install -U mlx-audio
```

## Environment Variables

For full functionality including script generation, create a `.env` file in the project root with:

```bash
GEMINI_API_KEY=your_gemini_api_key_here
SCRIPTER_MODEL=gemini-1.5-flash-latest
```

If these are not provided, the app will run in demo mode with basic TTS functionality (script generation disabled).

## Quick Start with Docker Compose

### Standard Version (Intel/AMD64 and Apple Silicon)
```bash
docker-compose up --build
```

### Apple Silicon Optimized Version
For the best performance on Apple Silicon devices:
```bash
docker-compose --profile apple-silicon up --build kokoro-streamer-m1
```

### Access the application:
Open your browser and go to `http://localhost:7860`

### Stop the application:
```bash
docker-compose down
```

## Manual Docker Build and Run

### Standard Build
```bash
docker build -t kokoro-streamer .
docker run -p 7860:7860 --env-file .env kokoro-streamer
```

### Apple Silicon Optimized Build
```bash
docker build -f Dockerfile.apple-silicon -t kokoro-streamer-m1 .
docker run -p 7860:7860 --env-file .env kokoro-streamer-m1
```

### Multi-platform Build (for distribution)
```bash
docker buildx build --platform linux/amd64,linux/arm64 -t kokoro-streamer .
```

## Features

- **🚀 Apple Silicon Optimization**: MLX-powered acceleration for M1/M2/M3 devices
- **Streaming Audio**: Real-time text-to-speech generation
- **Multiple Voices**: Choose from various voice options
- **Web Interface**: User-friendly Gradio interface
- **Script Generation**: AI-powered script enhancement (requires Gemini API)
- **Model Caching**: Persistent model downloads across container restarts
- **Auto-Detection**: Automatically uses the best pipeline for your hardware

## Performance Comparison

| Platform | Pipeline | Performance |
|----------|----------|-------------|
| Apple Silicon M1/M2/M3 | MLX Kokoro | **🚀 Optimized** - Up to 3x faster |
| Apple Silicon M1/M2/M3 | Standard Kokoro | Standard CPU performance |
| Intel/AMD CPU | Standard Kokoro | Standard CPU performance |
| NVIDIA GPU | Standard Kokoro | GPU acceleration |

## Configuration

### Environment Variables

- `GRADIO_SERVER_NAME`: Server hostname (default: 0.0.0.0)
- `GRADIO_SERVER_PORT`: Server port (default: 7860)
- `PYTHONUNBUFFERED`: Ensures Python output is not buffered
- `GEMINI_API_KEY`: Google Gemini API key for script generation
- `SCRIPTER_MODEL`: Gemini model to use (default: gemini-1.5-flash-latest)

### Volumes

The docker-compose setup includes a volume for Hugging Face model cache:
- `huggingface_cache`: Stores downloaded models to avoid re-downloading

## Troubleshooting

### Container won't start
Check the logs:
```bash
docker-compose logs kokoro-streamer
```

### Apple Silicon issues
If MLX installation fails, the app will automatically fall back to the standard pipeline:
```bash
# Check if you're on Apple Silicon
uname -m
# Should return: arm64

# Manually install MLX
pip install mlx-audio
```

### Port already in use
Change the port mapping in `docker-compose.yml`:
```yaml
ports:
  - "8080:7860"  # Use port 8080 instead
```

### Model download issues
On first run, the container will download the appropriate model:
- Standard: `hexgrad/Kokoro-82M` (~82MB)
- MLX: `mlx-community/Kokoro-82M-bf16` (~41MB, optimized)

### Script generation not working
Ensure your `.env` file contains valid `GEMINI_API_KEY`. The app will show "Demo Mode" if the API key is missing.

## Health Check

The container includes a health check that verifies the application is responding:
- Check interval: 30 seconds
- Timeout: 10 seconds
- Start period: 60 seconds (allows time for model loading)
- Retries: 3

Check health status:
```bash
docker-compose ps
```

## Performance Notes

### Apple Silicon (MLX)
- **CPU Usage**: Optimized for Apple Silicon Neural Engine
- **Memory**: Requires approximately 1GB RAM for MLX model
- **Storage**: MLX model is ~41MB (smaller than standard)
- **Speed**: Up to 3x faster than standard pipeline

### Standard Pipeline
- **CPU Usage**: Standard CPU/GPU acceleration
- **Memory**: Requires approximately 1-2GB RAM for model loading
- **Storage**: Standard model is ~82MB
- **Network**: First startup downloads models from Hugging Face Hub

## Security

- Runs as non-root user (`appuser`)
- Minimal base image (python:3.12-slim)
- Only necessary ports exposed
- Environment variables isolated in container

## Links

- [MLX Kokoro Model](https://huggingface.co/mlx-community/Kokoro-82M-bf16)
- [Original Kokoro Model](https://huggingface.co/hexgrad/Kokoro-82M)
- [MLX Audio Documentation](https://github.com/ml-explore/mlx-audio) 
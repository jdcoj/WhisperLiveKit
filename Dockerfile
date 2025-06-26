FROM pytorch/pytorch:2.7.1-cuda12.6-cudnn9-devel

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV CUDA_HOME=/usr/local/cuda
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/local/nvidia/lib64:$LD_LIBRARY_PATH
ENV PATH=/usr/local/cuda/bin:$PATH

WORKDIR /app

ARG EXTRAS
ARG HF_PRECACHE_DIR
ARG HF_TKN_FILE

# Install system dependencies and fix cuDNN
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        ffmpeg \
        git \
        portaudio19-dev \
        libsndfile1 \
        libsndfile1-dev \
        wget \
        gnupg \
        pciutils && \
    # 安裝正確的 cuDNN 庫
    wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb && \
    dpkg -i cuda-keyring_1.1-1_all.deb && \
    apt-get update && \
    apt-get install -y libcudnn9-cuda-12 libcudnn9-dev-cuda-12 && \
    # 創建符號連結確保庫文件可以被找到
    ln -sf /usr/lib/x86_64-linux-gnu/libcudnn*.so.9 /usr/local/cuda/lib64/ || true && \
    ldconfig && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    rm -f cuda-keyring_1.1-1_all.deb

# 驗證 CUDA 和 cuDNN 安裝
RUN python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}'); print(f'cuDNN version: {torch.backends.cudnn.version()}'); print(f'CUDA device count: {torch.cuda.device_count()}')" || echo "CUDA verification failed but continuing..."

# 安裝 diart
RUN pip install diart
RUN pip install mosestokenizer

COPY . .

# Install WhisperLiveKit directly, allowing for optional dependencies
RUN if [ -n "$EXTRAS" ]; then \
      echo "Installing with extras: [$EXTRAS]"; \
      pip install --no-cache-dir .[$EXTRAS]; \
    else \
      echo "Installing base package only"; \
      pip install --no-cache-dir .; \
    fi

# Enable in-container caching for Hugging Face models
VOLUME ["/root/.cache/huggingface/hub"]

# Conditionally copy a cache directory if provided
RUN if [ -n "$HF_PRECACHE_DIR" ]; then \
      echo "Copying Hugging Face cache from $HF_PRECACHE_DIR"; \
      mkdir -p /root/.cache/huggingface/hub && \
      cp -r $HF_PRECACHE_DIR/* /root/.cache/huggingface/hub; \
    else \
      echo "No local Hugging Face cache specified, skipping copy"; \
    fi

# Conditionally copy a Hugging Face token if provided
RUN if [ -n "$HF_TKN_FILE" ]; then \
      echo "Copying Hugging Face token from $HF_TKN_FILE"; \
      mkdir -p /root/.cache/huggingface && \
      cp $HF_TKN_FILE /root/.cache/huggingface/token; \
    else \
      echo "No Hugging Face token file specified, skipping token setup"; \
    fi
    
# Expose port for the transcription server
EXPOSE 8000

ENTRYPOINT ["whisperlivekit-server", "--host", "0.0.0.0"]

# Default args
CMD ["--model", "tiny.en"]
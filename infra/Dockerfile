FROM --platform=linux/amd64 ghcr.io/astral-sh/uv:latest AS uvbin
FROM --platform=linux/amd64 huggingface/transformers-pytorch-gpu:latest

WORKDIR /opt/app
RUN apt-get update && apt-get install -y git ffmpeg build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY --from=uvbin /uv /usr/local/bin/uv

COPY pyproject.toml ./
COPY uv.lock ./

RUN uv sync -v --python 3.11.12 --no-install-project

ENV VIRTUAL_ENV=/app/.venv
ENV PATH="${VIRTUAL_ENV}/bin:${PATH}"

COPY . .

ENV PATH="/opt/app/.venv/bin:${PATH}"
ENV CUDA_VISIBLE_DEVICES=""
ENTRYPOINT ["streamlit", "run", "main.py", "--server.port=8501", "--server.address=0.0.0.0"]

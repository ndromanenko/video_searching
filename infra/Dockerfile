FROM python:3.11-slim

ENV UV_PROJECT_ENVIRONMENT=/usr/local

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/ 

RUN mkdir -p /root/.ssh && \
    chmod 700 /root/.ssh && \
    ssh-keyscan github.com >> /root/.ssh/known_hosts


RUN --mount=type=ssh \ 
    --mount=type=bind,source=uv.lock,target=uv.lock \ 
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \ 
    uv sync -v --frozen --no-install-project --python 3.11.12

COPY ../pyproject.toml ../uv.lock ./

RUN --mount=type=ssh \ 
    uv sync --frozen --python 3.11.12

ENTRYPOINT ["streamlit", "run", "main.py", "--server.port=8501", "--server.address=0.0.0.0"]
# ENTRYPOINT ["tail", "-f", "/dev/null"]
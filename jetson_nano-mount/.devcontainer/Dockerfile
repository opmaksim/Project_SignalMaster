# Use NVIDIA L4T Base image
FROM nvcr.io/nvidia/l4t-base:r32.7.1

ARG DEBIAN_FRONTEND=noninteractive
ARG USER=vscode

# 1. Install sudo and create the vscode user with root privileges
RUN apt-get update && apt-get install -y --no-install-recommends sudo \
    && useradd -m -d /home/$USER -s /bin/bash $USER \
    && usermod -aG sudo $USER \
    && echo "$USER ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

# 2. Install essential dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential make \
    ca-certificates git \
    libssl-dev zlib1g-dev libbz2-dev \
    libreadline-dev libsqlite3-dev \
    wget curl llvm \
    libncurses5-dev xz-utils tk-dev \
    libxml2-dev libxmlsec1-dev libffi-dev \
    liblzma-dev

# 3. Switch to the vscode user
USER $USER
ARG HOME="/home/$USER"
ARG PYTHON_VERSION=3.12

# 4. Set environment variables for pyenv and PATH
ENV PYENV_ROOT="${HOME}/.pyenv"
ENV PATH="${PYENV_ROOT}/shims:${PYENV_ROOT}/bin:${HOME}/.local/bin:$PATH"

# 5. Install pyenv, Python, and poetry
RUN echo "Installing pyenv and poetry" \
    && curl https://pyenv.run | bash \
    && ${PYENV_ROOT}/bin/pyenv install ${PYTHON_VERSION} \
    && ${PYENV_ROOT}/bin/pyenv global ${PYTHON_VERSION} \
    && mkdir -p /home/$USER/.poetry-cache \
    && export TMPDIR=/home/$USER/.poetry-cache \
    && curl -sSL https://install.python-poetry.org | python3 - \
    && poetry config virtualenvs.in-project true

# 6. Set permissions to ensure the user can access installed tools
RUN sudo chown -R $USER:$USER /home/$USER

# Optional: Set the working directory
WORKDIR /home/$USER

# Optional: Expose ports or add entry points if needed
# EXPOSE 8000

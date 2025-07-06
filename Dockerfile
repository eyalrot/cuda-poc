FROM pytorch/pytorch:2.7.1-cuda12.8-cudnn9-devel

# Install build dependencies
RUN apt-get update && apt-get install -y \
    cmake \
    build-essential \
    libgtest-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /workspace

# Copy project files
COPY . .

# Build the project
RUN mkdir -p build && cd build && \
    cmake -DCMAKE_BUILD_TYPE=Release .. && \
    make -j$(nproc)

# Default command
CMD ["bash"]
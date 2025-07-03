FROM nvidia/cuda:12.9.0-devel-ubuntu22.04

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
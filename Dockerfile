FROM python:3.11-slim

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

# Set working directory
WORKDIR /app

# Copy project files
COPY pyproject.toml uv.lock* ./
COPY src/ ./src/
COPY examples/ ./examples/

# Install dependencies
RUN uv sync --frozen

# Expose gRPC port
EXPOSE 50051

# Set environment variables
ENV RAGFLOW_BASE_URL=http://ragflow:9380
ENV RAGFLOW_API_TOKEN=""

# Run the server
CMD ["uv", "run", "python", "-m", "src.server"]
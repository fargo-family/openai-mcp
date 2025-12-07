# syntax=docker/dockerfile:1.7
FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Copy only what we need to install dependencies
COPY src/requirements.txt ./src/requirements.txt
COPY external ./external
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r src/requirements.txt

# Copy the FastMCP application code
COPY src ./src

ENV PYTHONPATH=/app/src
EXPOSE 8000

CMD ["python", "-m", "openai_fastmcp"]

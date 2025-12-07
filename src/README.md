# FastMCP OpenAI Gateway

This package exposes OpenAI (and Azure OpenAI) Chat Completions, Image Generation, and Audio Generation endpoints as Model Context Protocol tools using [FastMCP](https://github.com/jlowin/fastmcp).

## Features

- MCP tool for chat completions with optional system prompt, JSON mode, sampling controls, and metadata passthrough
- MCP tool for DALL·E/gpt-image generations with selectable size, style, and output format (assets streamed via Azure Blob URLs)
- MCP tool for neural text-to-speech that uploads audio clips to Azure Blob Storage and returns shareable URLs
- MCP tool for video generation (Sora) that uploads MP4/thumbnail assets to Azure Blob Storage and returns shareable URLs
- MCP tool that explains which model/deployment backs each capability
- Mandatory API-key authentication enforced for every HTTP transport request
- Environment-variable driven configuration with first-class Azure OpenAI support
- Optional Azure Blob Storage persistence for image, audio, and video payloads

## Getting started

1. Copy `.env.example` to `.env` and fill in:
   ```bash
   cp .env.example .env
   ```
2. Install the dependencies (installs the vendored FastMCP + OpenAI clients from `external/`):
   ```bash
   pip install -r src/requirements.txt
   ```
3. Launch the server:
   ```bash
   python -m openai_fastmcp
   ```

By default the server listens on `0.0.0.0:8000` and requires clients to send `Authorization: Bearer <MCP_SERVER_API_KEY>` headers when connecting over HTTP/SSE/streamable HTTP.

### Docker

The included `Dockerfile` builds a minimal production image. Providing a `.env` file at runtime keeps secrets out of the image itself:

```bash
docker build -t fastmcp-openai .
docker run --env-file .env -p 8000:8000 fastmcp-openai
```

### Azure OpenAI

Set `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_API_KEY`, and (optionally) `AZURE_OPENAI_API_VERSION` to automatically switch the client to Azure mode. Deployment-specific overrides (`AZURE_OPENAI_*_DEPLOYMENT`) let you point chat, image, and audio calls to different resource deployments.

### Azure Blob Storage uploads

Set the following environment variables to upload generated images, audio clips, and videos to Azure Blob Storage. The `generate_image`, `synthesize_speech`, and `generate_video` tools **only** return `blob_url` entries—raw base64 payloads are no longer included—so these settings are required to retrieve media outputs.

| Variable | Required | Description |
| --- | --- | --- |
| `AZURE_STORAGE_CONNECTION_STRING` | ✅ | Connection string for the target storage account. |
| `AZURE_BLOB_CONTAINER` | ✅ | Container that should receive the uploaded media blobs. Must already exist. |
| `AZURE_BLOB_PUBLIC_BASE_URL` | ✅ | Public `https://` base URL for the container (for example, `https://myacct.blob.core.windows.net/public`). |
| `AZURE_BLOB_PATH_PREFIX` | ❌ | Optional path prefix (e.g., `fastmcp`) that is prepended before the media-type folders. |
| `AZURE_BLOB_IMAGE_ROOT` | ❌ | Override the root folder for image outputs (defaults to `images`). |
| `AZURE_BLOB_AUDIO_ROOT` | ❌ | Override the root folder for audio outputs (defaults to `audio`). |
| `AZURE_BLOB_VIDEO_ROOT` | ❌ | Override the root folder for video outputs (defaults to `videos`). |

Blobs are uploaded under `<prefix>/<type-root>/<timestamp>-<uuid>.<ext>` (for example, `fastmcp/images/20241204-...png`). Set the optional root overrides to redirect a single media type to a custom folder. `Content-Type` headers are set automatically so the assets can be served directly to browsers. If these settings are omitted the media tools raise a runtime error to remind you to configure storage.

## MCP Tools

| Tool name | Description |
| --- | --- |
| `chat_completion` | Accesses the Chat Completions API with optional system prompt, sampling controls, and JSON mode. |
| `generate_image` | Calls the Images API and returns Azure Blob URLs for each variation (no inline base64). |
| `synthesize_speech` | Uses the Audio Speech API, uploads the clip to Azure Blob Storage, and returns its URL. |
| `generate_video` | Uses the Videos API (Sora), uploads the requested asset to Azure Blob Storage, and returns its URL. |
| `list_supported_models` | Returns the configured model/deployment for chat, image, audio, and video tools. |

All tools are synchronous from the caller's perspective but execute via FastMCP's worker pool to avoid blocking the event loop.

> **Note:** OpenAI's video generation API is not currently available via Azure OpenAI. The `generate_video` tool will raise a runtime error when the server is configured for Azure.

## Azure DevOps pipeline

The root-level `azure-pipelines.yml` definition builds the Docker image and pushes it to Docker Hub. Configure two pipeline variables (or variable group entries) before running the pipeline:

- `DOCKERHUB_SERVICE_CONNECTION`: name of the Azure DevOps Docker Hub service connection.
- `DOCKERHUB_REPOSITORY`: full repository name, e.g. `my-org/fastmcp-openai`.

Every merge to `main` triggers the pipeline, which checks out submodules, builds the container using the provided `Dockerfile`, and publishes both a `latest` tag and a build-number tag.

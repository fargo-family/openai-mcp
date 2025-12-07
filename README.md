# OpenAI FastMCP Gateway

This repository wraps the official `openai` Python SDK and `fastmcp` runtime (vendored under `external/`) to expose three MCP tools—chat completions, image generation, and audio synthesis—over authenticated HTTP/SSE transports.

## Project layout

- `external/`: mirrors the upstream `openai/openai-python` and `jlowin/fastmcp` sources.
- `src/openai_fastmcp/`: custom FastMCP server, configuration loader, and OpenAI service wrapper.
- `Dockerfile`: production image that installs the vendored dependencies and runs `python -m openai_fastmcp`.
- `azure-pipelines.yml`: CI definition that builds/pushes the container to Docker Hub via Azure DevOps.

## Setup

1. Copy the sample environment file and edit secrets:
   ```bash
   cp .env.example .env
   ```
2. Install dependencies (installs from the local `external/` tree):
   ```bash
   pip install -r src/requirements.txt
   ```
3. Run the server:
   ```bash
   python -m openai_fastmcp
   ```

The FastMCP server listens on `0.0.0.0:8000` by default and **requires** every request to include `Authorization: Bearer $MCP_SERVER_API_KEY`.

## Docker

```bash
docker build -t fastmcp-openai .
docker run --env-file .env -p 8000:8000 fastmcp-openai
```

## Azure OpenAI

Set `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_API_KEY`, and (optionally) `AZURE_OPENAI_API_VERSION`. Per-capability deployment overrides are available via:
- `AZURE_OPENAI_CHAT_DEPLOYMENT`
- `AZURE_OPENAI_IMAGE_DEPLOYMENT`
- `AZURE_OPENAI_AUDIO_DEPLOYMENT`

Leaving those unset falls back to the generic `OPENAI_*_MODEL` defaults.

## Azure Blob Storage

Media-producing tools (`generate_image`, `synthesize_speech`, `generate_video`) upload their outputs to Azure Blob Storage and return only the resulting public blob URLs—raw base64 payloads are not included. Configure the following environment variables (matching the ones described in `src/README.md`) so the server can store and share these assets:

- `AZURE_STORAGE_CONNECTION_STRING`
- `AZURE_BLOB_CONTAINER`
- `AZURE_BLOB_PUBLIC_BASE_URL`
- (Optional) `AZURE_BLOB_PATH_PREFIX`, `AZURE_BLOB_IMAGE_ROOT`, `AZURE_BLOB_AUDIO_ROOT`, `AZURE_BLOB_VIDEO_ROOT`

If these variables are missing, the media tools raise a runtime error reminding you to configure storage.

## MCP tools

| Tool | Purpose |
| --- | --- |
| `chat_completion` | Chat Completions API with optional system prompt, JSON mode, sampling knobs, metadata passthrough. |
| `generate_image` | gpt-image/DALL·E style image generation. Uploads assets to Azure Blob Storage and returns shareable URLs. |
| `synthesize_speech` | Text-to-speech using the Audio Speech endpoint. Uploads audio clips to Azure Blob Storage and returns shareable URLs. |
| `generate_video` | Short-form video generation (Sora). Uploads MP4/thumbnail assets to Azure Blob Storage and returns shareable URLs. |
| `list_supported_models` | Reports the configured model/deployment used for each capability (chat/image/audio/video). |

> Video generation is available only when using api.openai.com (Sora). Azure OpenAI currently lacks the `/videos` endpoints, so the tool raises a runtime error in Azure mode.

## Calling the server from `mcp.json`

When registering this server inside an agent client, include the header in the transport definition. Example snippet:

```json
{
  "servers": {
    "fastmcp-openai": {
      "transport": {
        "type": "sse",
        "url": "http://localhost:8000/mcp",
        "headers": {
          "Authorization": "Bearer ${env:MCP_SERVER_API_KEY}"
        }
      }
    }
  }
}
```

Most MCP clients (Claude Desktop, VS Code MCP, etc.) support `${env:VAR}` expansion; otherwise, replace the value with your actual shared secret. If you connect over Streamable HTTP, add the same `headers` block inside that transport configuration.

## Pipeline

`azure-pipelines.yml` checks out submodules, builds the Docker image, and pushes both `latest` and build-specific tags to Docker Hub. Provide two pipeline variables/secrets:
- `DOCKERHUB_SERVICE_CONNECTION`: name of the Docker registry service connection.
- `DOCKERHUB_REPOSITORY`: e.g. `my-org/fastmcp-openai`.

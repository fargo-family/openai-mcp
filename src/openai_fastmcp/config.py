from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Literal

from dotenv import load_dotenv

# Automatically load variables from a local .env file if present
load_dotenv()

Provider = Literal["openai", "azure"]


@dataclass(frozen=True, slots=True)
class ServerConfig:
    """Runtime configuration for the FastMCP server."""

    name: str
    instructions: str
    host: str
    port: int
    api_key: str
    base_url: str | None
    required_scopes: list[str]


@dataclass(frozen=True, slots=True)
class OpenAIConfig:
    """Configuration for either OpenAI or Azure OpenAI."""

    provider: Provider
    api_key: str
    base_url: str | None
    organization: str | None
    azure_endpoint: str | None
    azure_api_version: str | None
    azure_ad_token: str | None
    chat_model: str
    image_model: str
    audio_model: str
    default_voice: str
    video_model: str


@dataclass(frozen=True, slots=True)
class AzureBlobStorageConfig:
    """Optional configuration for persisting media artifacts in Azure Blob Storage."""

    connection_string: str
    container: str
    public_base_url: str
    path_prefix: str | None
    image_root: str
    audio_root: str
    video_root: str


@dataclass(frozen=True, slots=True)
class Settings:
    server: ServerConfig
    openai: OpenAIConfig
    storage: AzureBlobStorageConfig | None = None

    @classmethod
    def from_env(cls) -> "Settings":
        """Load settings from environment variables (with sane defaults)."""

        server = _load_server_config()
        openai = _load_openai_config()
        storage = _load_blob_storage_config()
        return cls(server=server, openai=openai, storage=storage)


def _load_server_config() -> ServerConfig:
    api_key = _require("MCP_SERVER_API_KEY")
    host = os.getenv("MCP_SERVER_HOST", "0.0.0.0")
    port = _as_int(os.getenv("MCP_SERVER_PORT"), default=8000, name="MCP_SERVER_PORT")
    base_url = os.getenv("MCP_SERVER_BASE_URL")
    scopes = _as_scopes(os.getenv("MCP_REQUIRED_SCOPES"))

    name = os.getenv("MCP_SERVER_NAME", "OpenAI + Azure MCP Gateway")
    instructions = os.getenv(
        "MCP_SERVER_INSTRUCTIONS",
        "Expose OpenAI chat, image, and audio generation endpoints with a single MCP server.",
    )

    return ServerConfig(
        name=name,
        instructions=instructions,
        host=host,
        port=port,
        api_key=api_key,
        base_url=base_url,
        required_scopes=scopes,
    )


def _load_openai_config() -> OpenAIConfig:
    use_azure = bool(os.getenv("AZURE_OPENAI_ENDPOINT"))

    default_voice = os.getenv("OPENAI_DEFAULT_VOICE", "alloy")

    if use_azure:
        api_key = _require("AZURE_OPENAI_API_KEY")
        endpoint = _require("AZURE_OPENAI_ENDPOINT")
        api_version = os.getenv("AZURE_OPENAI_API_VERSION") or os.getenv(
            "OPENAI_API_VERSION", "2024-10-21"
        )

        chat_model = _value_with_fallback(
            "AZURE_OPENAI_CHAT_DEPLOYMENT",
            os.getenv("AZURE_OPENAI_DEPLOYMENT")
            or os.getenv("OPENAI_CHAT_MODEL"),
            default="gpt-4o-mini",
        )
        image_model = _value_with_fallback(
            "AZURE_OPENAI_IMAGE_DEPLOYMENT",
            os.getenv("AZURE_OPENAI_DEPLOYMENT")
            or os.getenv("OPENAI_IMAGE_MODEL"),
            default="gpt-image-1",
        )
        audio_model = _value_with_fallback(
            "AZURE_OPENAI_AUDIO_DEPLOYMENT",
            os.getenv("AZURE_OPENAI_DEPLOYMENT")
            or os.getenv("OPENAI_AUDIO_MODEL"),
            default="gpt-4o-mini-tts",
        )

        return OpenAIConfig(
            provider="azure",
            api_key=api_key,
            base_url=None,
            organization=None,
            azure_endpoint=endpoint,
            azure_api_version=api_version,
            azure_ad_token=os.getenv("AZURE_OPENAI_AD_TOKEN"),
            chat_model=chat_model,
            image_model=image_model,
            audio_model=audio_model,
            default_voice=default_voice,
            video_model=_value_with_fallback(
                "AZURE_OPENAI_VIDEO_DEPLOYMENT",
                os.getenv("AZURE_OPENAI_DEPLOYMENT")
                or os.getenv("OPENAI_VIDEO_MODEL"),
                default="sora-2",
            ),
        )

    api_key = _require("OPENAI_API_KEY")
    return OpenAIConfig(
        provider="openai",
        api_key=api_key,
        base_url=os.getenv("OPENAI_BASE_URL"),
        organization=os.getenv("OPENAI_ORG"),
        azure_endpoint=None,
        azure_api_version=None,
        azure_ad_token=None,
        chat_model=os.getenv("OPENAI_CHAT_MODEL", "gpt-4.1-mini"),
        image_model=os.getenv("OPENAI_IMAGE_MODEL", "gpt-image-1"),
        audio_model=os.getenv("OPENAI_AUDIO_MODEL", "gpt-4o-mini-tts"),
        default_voice=default_voice,
        video_model=os.getenv("OPENAI_VIDEO_MODEL", "sora-2"),
    )


def _load_blob_storage_config() -> AzureBlobStorageConfig | None:
    connection_string = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
    container = os.getenv("AZURE_BLOB_CONTAINER")
    public_base_url = os.getenv("AZURE_BLOB_PUBLIC_BASE_URL")

    if not (connection_string and container and public_base_url):
        return None

    prefix = os.getenv("AZURE_BLOB_PATH_PREFIX")
    normalized_prefix = _normalize_path_segment(prefix)

    image_root = _normalize_path_segment(os.getenv("AZURE_BLOB_IMAGE_ROOT")) or "images"
    audio_root = _normalize_path_segment(os.getenv("AZURE_BLOB_AUDIO_ROOT")) or "audio"
    video_root = _normalize_path_segment(os.getenv("AZURE_BLOB_VIDEO_ROOT")) or "videos"

    return AzureBlobStorageConfig(
        connection_string=connection_string,
        container=container,
        public_base_url=public_base_url.rstrip("/"),
        path_prefix=normalized_prefix,
        image_root=image_root,
        audio_root=audio_root,
        video_root=video_root,
    )


def _require(name: str) -> str:
    value = os.getenv(name)
    if value:
        return value
    raise RuntimeError(f"Missing required environment variable: {name}")


def _as_int(raw: str | None, *, default: int, name: str) -> int:
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise RuntimeError(f"{name} must be an integer (received: {raw!r})") from exc


def _as_scopes(raw: str | None) -> list[str]:
    if not raw:
        return []
    separators = [",", " "]
    for sep in separators:
        raw = raw.replace(sep, " ")
    return [scope.strip() for scope in raw.split(" ") if scope.strip()]


def _value_with_fallback(primary_var: str, fallback: str | None, *, default: str) -> str:
    primary = os.getenv(primary_var)
    if primary and primary.strip():
        return primary.strip()
    if fallback and fallback.strip():
        return fallback.strip()
    return default


def _normalize_path_segment(raw: str | None) -> str | None:
    if not raw:
        return None
    trimmed = raw.strip().strip("/")
    if not trimmed:
        return None
    parts = [segment for segment in trimmed.split("/") if segment]
    return "/".join(parts) if parts else None

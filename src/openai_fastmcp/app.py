from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from enum import StrEnum
from typing import Annotated, Any

from fastmcp import FastMCP
from pydantic import Field
from starlette.requests import Request
from starlette.responses import PlainTextResponse

from .auth import StaticAPIKeyAuth
from .config import Settings
from .openai_service import OpenAIService


_LOGGING_CONFIGURED = False


def configure_logging(level: int = logging.DEBUG) -> None:
    """Turn on verbose logging for the entire process once per runtime."""

    global _LOGGING_CONFIGURED
    if _LOGGING_CONFIGURED:
        return

    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    noisy_loggers = (
        "fastmcp",
        "httpx",
        "httpcore",
        "openai",
    )
    for name in noisy_loggers:
        logging.getLogger(name).setLevel(level)

    _LOGGING_CONFIGURED = True


class ImageSize(StrEnum):
    """Supported image output resolutions."""

    SQUARE_1024 = "1024x1024"
    PORTRAIT_1024x1536 = "1024x1536"
    LANDSCAPE_1536x1024 = "1536x1024"
    AUTO = "auto"


class ImageQuality(StrEnum):
    """Supported image quality presets."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    AUTO = "auto"


class VideoAssetVariant(StrEnum):
    """Available downloadable video asset variants."""

    VIDEO = "video"
    THUMBNAIL = "thumbnail"
    SPRITESHEET = "spritesheet"


class ModelCapability(StrEnum):
    """Capability filters recognized by the supported-models tool."""

    CHAT = "chat"
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"



def create_server(settings: Settings | None = None) -> FastMCP:
    """Instantiate the FastMCP server with all OpenAI tools registered."""

    configure_logging()
    settings = settings or Settings.from_env()
    service = OpenAIService(settings.openai, storage=settings.storage)

    auth = StaticAPIKeyAuth(
        api_key=settings.server.api_key,
        base_url=settings.server.base_url,
        required_scopes=settings.server.required_scopes,
    )

    @asynccontextmanager
    async def lifespan(_: FastMCP):
        try:
            yield {}
        finally:
            await service.aclose()

    server = FastMCP(
        name=settings.server.name,
        instructions=settings.server.instructions,
        auth=auth,
        lifespan=lifespan,
    )

    @server.tool
    async def chat_completion(
        prompt: Annotated[
            str,
            Field(description="Primary user content to send to the chat model."),
        ],
        *,
        system_prompt: Annotated[
            str | None,
            Field(
                description="Optional system instruction injected ahead of the user message to steer tone, policy, or persona.",
            ),
        ] = None,
        model: Annotated[
            str | None,
            Field(description="Override the default chat model (e.g., gpt-4.1-mini)."),
        ] = None,
        temperature: Annotated[
            float,
            Field(description="Softmax temperature; lower values are more deterministic."),
        ] = 0.2,
        top_p: Annotated[
            float,
            Field(description="Nucleus sampling cutoff applied alongside temperature."),
        ] = 1.0,
        max_output_tokens: Annotated[
            int | None,
            Field(description="Hard limit on assistant tokens; defaults to model maximum when omitted."),
        ] = None,
        user: Annotated[
            str | None,
            Field(description="Opaque user identifier forwarded to OpenAI for abuse monitoring."),
        ] = None,
        response_format: Annotated[
            str | None,
            Field(description="Set to 'json' or a named schema alias to force structured output."),
        ] = None,
        presence_penalty: Annotated[
            float,
            Field(description="Penalty encouraging the model to introduce new topics."),
        ] = 0.0,
        frequency_penalty: Annotated[
            float,
            Field(description="Penalty discouraging repeated tokens."),
        ] = 0.0,
        seed: Annotated[
            int | None,
            Field(description="Determinism seed; identical inputs + seed yield identical outputs."),
        ] = None,
        metadata: Annotated[
            dict[str, Any] | None,
            Field(description="Arbitrary metadata forwarded to OpenAI for audit tagging."),
        ] = None,
    ) -> dict[str, Any]:
        """Call OpenAI chat completions.

        This tool wraps the `/chat/completions` API and streams nothing— it waits
        for the model to finish and returns the first choice. Provide an optional
        `system_prompt` when you need to steer tone or inject instructions that
        should not be exposed to the end user. Sampling controls (`temperature`,
        `top_p`, `presence_penalty`, `frequency_penalty`) mirror the native API
        parameters and default to conservative values so the responses remain
        deterministic unless you override them. Use `response_format="json"` to
        force JSON Schema compliant output. The response returns the final text,
        finish reason, model id, raw OpenAI payload, and usage statistics to help
        the LLM reason about token budgeting.
        """

        return await service.chat_completion(
            prompt,
            system_prompt=system_prompt,
            model=model,
            temperature=temperature,
            top_p=top_p,
            max_output_tokens=max_output_tokens,
            user=user,
            response_format=response_format,
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty,
            seed=seed,
            metadata=metadata,
        )

    @server.tool
    async def generate_image(
        prompt: Annotated[
            str,
            Field(description="Natural language description of the image to render."),
        ],
        *,
        size: Annotated[
            ImageSize,
            Field(description="Resolution enum: 1024x1024, 1024x1536, 1536x1024, or auto."),
        ] = ImageSize.SQUARE_1024,
        quality: Annotated[
            ImageQuality,
            Field(description="Image fidelity preset: low, medium, high, or auto."),
        ] = ImageQuality.HIGH,
        count: Annotated[
            int,
            Field(description="Number of variations to render (1-10)."),
        ] = 1,
        user: Annotated[
            str | None,
            Field(description="Opaque identifier forwarded to OpenAI for abuse tracking."),
        ] = None,
    ) -> dict[str, Any]:
        """Generate images (gpt-image-1 / DALL·E).

        This tool calls `images.generate`, uploads each variation to Azure Blob
        Storage, and returns the publicly accessible blob URLs. `size` is a
        StrEnum covering `1024x1024`, `1024x1536`, `1536x1024`, or `auto`;
        `quality` is a StrEnum covering `low`, `medium`, `high`, or `auto`;
        `count` controls how many variations (1-10). The response includes the
        model name, creation timestamp, and an array with `blob_url` entries
        (plus any OpenAI `revised_prompt` annotations).
        """

        if not 1 <= count <= 10:
            raise ValueError("count must be between 1 and 10")

        return await service.generate_image(
            prompt,
            size=size,
            quality=quality,
            count=count,
            user=user,
        )

    @server.tool
    async def synthesize_speech(
        text: Annotated[
            str,
            Field(description="Plain text that should be converted into audio."),
        ],
        *,
        model: Annotated[
            str | None,
            Field(description="Override the default TTS model."),
        ] = None,
        voice: Annotated[
            str | None,
            Field(description="Voice preset such as 'alloy', 'verse', or Azure-specific names."),
        ] = None,
        response_format: Annotated[
            str,
            Field(description="Audio container to return: mp3, wav, flac, opus, etc."),
        ] = "mp3",
        speed: Annotated[
            float,
            Field(description="Playback rate multiplier where 1.0 is real-time."),
        ] = 1.0,
    ) -> dict[str, Any]:
        """Synthesize speech via the `/audio/speech` API.

        Provide plain text, optionally pick an alternate TTS `voice`, `model`,
        or `response_format` (`mp3`, `wav`, `flac`, `opus`). The server uploads
        the rendered audio to Azure Blob Storage and returns the blob URL along
        with the effective format, selected voice, and byte length.
        """

        return await service.generate_audio(
            text,
            model=model,
            voice=voice,
            response_format=response_format,
            speed=speed,
        )

    @server.tool
    async def generate_video(
        prompt: Annotated[
            str,
            Field(description="Rich description of the motion scene to synthesize."),
        ],
        *,
        model: Annotated[
            str | None,
            Field(description="Override the default Sora deployment name."),
        ] = None,
        seconds: Annotated[
            int,
            Field(description="Clip duration; must be 4, 8, or 12 seconds."),
        ] = 4,
        size: Annotated[
            str,
            Field(description="Output resolution such as '720x1280' (portrait) or '1280x720'."),
        ] = "720x1280",
        variant: Annotated[
            VideoAssetVariant,
            Field(description="Download target: 'video' (MP4), 'thumbnail', or 'spritesheet'."),
        ] = VideoAssetVariant.VIDEO,
    ) -> dict[str, Any]:
        """Create a short-form video clip using Sora.

        This tool submits a `/videos` job, polls until completion, and downloads
        the requested asset variant (`video` for MP4, `thumbnail`, or
        `spritesheet`). `seconds` must be 4, 8, or 12 and `size` controls the
        output resolution (e.g., `720x1280` portrait or `1280x720` landscape).
        Azure OpenAI does not expose Sora; when the server is configured for
        Azure this tool raises a runtime error so the caller can fall back.
        Returns the job metadata, including the video id, status, model, size,
        duration, variant, and the Azure `blob_url` pointing at the requested
        downloadable asset.
        """

        allowed_seconds = {4, 8, 12}
        if seconds not in allowed_seconds:
            raise ValueError("seconds must be one of 4, 8, or 12")

        allowed_variants = {"video", "thumbnail", "spritesheet"}
        if variant not in allowed_variants:
            raise ValueError("variant must be video, thumbnail, or spritesheet")

        return await service.generate_video(
            prompt,
            model=model,
            seconds=seconds,
            size=size,
            variant=variant,
        )

    @server.tool
    async def list_supported_models(
        capability: Annotated[
            ModelCapability | None,
            Field(
                description="Optional capability filter: chat, image, audio, or video.",
            ),
        ] = None,
        include_provider_metadata: Annotated[
            bool,
            Field(description="Append provider info (base URLs, endpoints) when true."),
        ] = True,
    ) -> dict[str, Any]:
        """Report configured models/deployments per capability.

        Use this tool to understand what models are available for use with the other tools on this server. it is HIGHLY recommended to call this tool at the start of any session to help the LLM reason about its capabilities and budget. Optionally provide a `capability` filter (chat, image, audio, or video) to narrow the response. Set `include_provider_metadata` to true to append provider details like base URLs and endpoints.
        """

        if isinstance(capability, ModelCapability):
            normalized = capability.value
        elif isinstance(capability, str):
            normalized = capability.lower()
        else:
            normalized = None
        return await service.list_supported_models(
            normalized,
            include_provider_metadata,
        )

    @server.custom_route("/healthz", methods=["GET"], include_in_schema=False)
    async def health_check(_: Request) -> PlainTextResponse:
        """Simple unauthenticated liveness probe."""

        return PlainTextResponse("ok", status_code=200)

    return server


def run() -> None:
    """Bootstrap the FastMCP HTTP server."""

    settings = Settings.from_env()
    server = create_server(settings)
    server.run(
        "http",
        host=settings.server.host,
        port=settings.server.port,
    )

from __future__ import annotations

import base64
import binascii
import datetime
import logging
import uuid
from typing import Any, Iterable

from azure.storage.blob import ContentSettings
from azure.storage.blob.aio import BlobServiceClient
from openai import AsyncAzureOpenAI, AsyncOpenAI
from openai.types.video import Video

from .config import AzureBlobStorageConfig, OpenAIConfig


logger = logging.getLogger(__name__)
_DEFAULT_CONTENT_TYPE = "application/octet-stream"
_STORAGE_REQUIRED_MSG = (
    "Azure Blob Storage uploads are not configured. Set AZURE_STORAGE_CONNECTION_STRING, "
    "AZURE_BLOB_CONTAINER, and AZURE_BLOB_PUBLIC_BASE_URL to enable asset URLs."
)
_AUDIO_CONTENT_TYPES: dict[str, str] = {
    "mp3": "audio/mpeg",
    "wav": "audio/wav",
    "flac": "audio/flac",
    "opus": "audio/ogg",
    "ogg": "audio/ogg",
    "aac": "audio/aac",
    "m4a": "audio/mp4",
}


class OpenAIService:
    """Wrapper that exposes high-level helpers for OpenAI and Azure OpenAI."""

    _CAPABILITY_TO_FIELD = {
        "chat": "chat_model",
        "image": "image_model",
        "audio": "audio_model",
        "video": "video_model",
    }

    def __init__(self, config: OpenAIConfig, *, storage: AzureBlobStorageConfig | None = None) -> None:
        self.config = config
        self.client = self._build_client(config)
        self._blob_uploader = AzureBlobUploader(storage) if storage else None

    async def chat_completion(
        self,
        prompt: str,
        *,
        system_prompt: str | None,
        model: str | None,
        temperature: float,
        top_p: float,
        max_output_tokens: int | None,
        user: str | None,
        response_format: str | None,
        presence_penalty: float,
        frequency_penalty: float,
        seed: int | None,
        metadata: dict[str, Any] | None,
    ) -> dict[str, Any]:
        """Execute a chat completion call and normalize the output."""

        final_model = model or self.config.chat_model
        messages = self._build_messages(prompt, system_prompt)

        response_format_param = None
        if response_format:
            if response_format.lower() == "json":
                response_format_param = {"type": "json_object"}
            else:
                response_format_param = {"type": response_format}

        completion = await self.client.chat.completions.create(
            model=final_model,
            messages=messages,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_output_tokens,
            response_format=response_format_param,
            user=user,
            seed=seed,
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty,
            metadata=metadata,
        )

        choice = completion.choices[0]
        text = self._extract_text(choice.message.content)

        return {
            "text": text,
            "finish_reason": choice.finish_reason,
            "model": completion.model,
            "usage": completion.usage.model_dump() if completion.usage else None,
            "raw": completion.model_dump(mode="json"),
        }

    async def generate_image(
        self,
        prompt: str,
        *,
        size: str,
        quality: str,
        count: int,
        user: str | None,
    ) -> dict[str, Any]:
        """Call the Images API and return normalized payload."""

        final_model = self.config.image_model

        normalized_quality = quality.lower()
        legacy_aliases = {
            "standard": "medium",
            "hd": "high",
        }
        normalized_quality = legacy_aliases.get(normalized_quality, normalized_quality)

        allowed_qualities = {"low", "medium", "high", "auto"}
        if normalized_quality not in allowed_qualities:
            allowed_str = ", ".join(sorted(allowed_qualities))
            raise ValueError(
                f"quality must be one of {allowed_str} (received: {quality!r})"
            )

        normalized_size = size.lower()
        allowed_sizes = {"1024x1024", "1024x1536", "1536x1024", "auto"}
        if normalized_size not in allowed_sizes:
            allowed_size_str = ", ".join(sorted(allowed_sizes))
            raise ValueError(
                f"size must be one of {allowed_size_str} (received: {size!r})"
            )

        response = await self.client.images.generate(
            model=final_model,
            prompt=prompt,
            size=normalized_size,
            quality=normalized_quality,
            n=count,
            user=user,
        )

        images = []
        for item in response.data:
            blob_url = self._require_blob_url(
                await self._persist_base64_media(
                    base64_payload=item.b64_json,
                    category="images",
                    extension="png",
                    content_type="image/png",
                ),
                asset_type="image",
            )

            images.append(
                {
                    "blob_url": blob_url,
                    "revised_prompt": getattr(item, "revised_prompt", None),
                }
            )

        return {
            "model": getattr(response, "model", final_model),
            "created": response.created,
            "images": images,
        }

    async def generate_audio(
        self,
        text: str,
        *,
        model: str | None,
        voice: str | None,
        response_format: str,
        speed: float,
    ) -> dict[str, Any]:
        """Convert text into speech and return a blob URL pointing to the audio asset."""

        final_model = model or self.config.audio_model
        final_voice = voice or self.config.default_voice

        speech = await self.client.audio.speech.create(
            model=final_model,
            voice=final_voice,
            input=text,
            response_format=response_format,
            speed=speed,
        )

        audio_bytes = await speech.aread()
        blob_url = self._require_blob_url(
            await self._persist_media_bytes(
                payload=audio_bytes,
                category="audio",
                extension=response_format,
                content_type=self._audio_content_type(response_format),
            ),
            asset_type="audio",
        )

        return {
            "voice": final_voice,
            "format": response_format,
            "byte_length": len(audio_bytes),
            "blob_url": blob_url,
        }

    async def generate_video(
        self,
        prompt: str,
        *,
        model: str | None,
        seconds: int,
        size: str,
        variant: str,
    ) -> dict[str, Any]:
        """Generate a short-form video clip and return a blob URL for the asset."""

        if self.config.provider == "azure":
            raise RuntimeError("Azure OpenAI does not currently support video generation")

        final_model = model or self.config.video_model

        video_job: Video = await self.client.videos.create_and_poll(
            prompt=prompt,
            model=final_model,
            seconds=seconds,
            size=size,
        )

        if video_job.status != "completed":
            raise RuntimeError(
                f"Video job {video_job.id} did not complete successfully (status={video_job.status})"
            )

        binary = await self.client.videos.download_content(video_job.id, variant=variant)
        video_bytes = await binary.aread()
        extension, content_type = self._video_variant_meta(variant)
        blob_url = self._require_blob_url(
            await self._persist_media_bytes(
                payload=video_bytes,
                category="videos",
                extension=extension,
                content_type=content_type,
            ),
            asset_type="video",
        )

        return {
            "video_id": video_job.id,
            "status": video_job.status,
            "model": video_job.model,
            "size": video_job.size,
            "seconds": video_job.seconds,
            "variant": variant,
            "byte_length": len(video_bytes),
            "blob_url": blob_url,
        }

    async def list_supported_models(
        self,
        capability: str | None,
        include_provider_metadata: bool,
    ) -> dict[str, Any]:
        """Return configured model/deployment info for each capability."""

        normalized = capability.lower() if capability else None
        if normalized and normalized not in self._CAPABILITY_TO_FIELD:
            raise ValueError(
                "capability must be one of chat, image, audio, video"
            )

        payload: dict[str, Any] = {}
        for cap, field_name in self._CAPABILITY_TO_FIELD.items():
            if normalized and cap != normalized:
                continue

            model_name = getattr(self.config, field_name, None)
            if not model_name:
                continue

            entry: dict[str, Any] = {
                "configured_model": model_name,
                "provider": self.config.provider,
                "azure_supported": self.config.provider == "azure" and cap != "video",
            }

            if cap == "video":
                entry["notes"] = (
                    "Video generation requires api.openai.com; Azure deployments do not support /videos."
                )

            payload[cap] = entry

        if include_provider_metadata:
            payload["_provider"] = {
                "provider": self.config.provider,
                "base_url": self.config.base_url,
                "azure_endpoint": self.config.azure_endpoint,
            }

        return payload

    async def aclose(self) -> None:
        await self.client.aclose()
        if self._blob_uploader:
            await self._blob_uploader.aclose()

    async def _persist_media_bytes(
        self,
        *,
        payload: bytes | None,
        category: str,
        extension: str,
        content_type: str | None,
    ) -> str | None:
        if not payload:
            return None
        if not self._blob_uploader:
            raise RuntimeError(_STORAGE_REQUIRED_MSG)

        sanitized_extension = (extension.lstrip(".") or "bin").lower()
        try:
            return await self._blob_uploader.upload(
                data=payload,
                category=category,
                extension=sanitized_extension,
                content_type=content_type or _DEFAULT_CONTENT_TYPE,
            )
        except Exception:  # pragma: no cover - defensive logging
            logger.exception("Failed to upload %s artifact to Azure Blob Storage", category)
            return None

    async def _persist_base64_media(
        self,
        *,
        base64_payload: str | None,
        category: str,
        extension: str,
        content_type: str | None,
    ) -> str | None:
        if not base64_payload:
            return None

        try:
            decoded = base64.b64decode(base64_payload)
        except (binascii.Error, ValueError):  # pragma: no cover - invalid base64
            logger.exception("Received invalid base64 payload for %s artifact", category)
            return None

        return await self._persist_media_bytes(
            payload=decoded,
            category=category,
            extension=extension,
            content_type=content_type,
        )

    @staticmethod
    def _require_blob_url(blob_url: str | None, *, asset_type: str) -> str:
        if blob_url:
            return blob_url
        raise RuntimeError(
            f"Unable to produce a publicly accessible {asset_type} asset URL. "
            f"{_STORAGE_REQUIRED_MSG}"
        )

    @staticmethod
    def _audio_content_type(response_format: str) -> str:
        return _AUDIO_CONTENT_TYPES.get(response_format.lower(), _DEFAULT_CONTENT_TYPE)

    @staticmethod
    def _video_variant_meta(variant: str) -> tuple[str, str]:
        normalized = variant.lower()
        mapping: dict[str, tuple[str, str]] = {
            "video": ("mp4", "video/mp4"),
            "thumbnail": ("png", "image/png"),
            "spritesheet": ("json", "application/json"),
        }
        return mapping.get(normalized, ("bin", _DEFAULT_CONTENT_TYPE))

    @staticmethod
    def _build_client(config: OpenAIConfig) -> AsyncOpenAI:
        if config.provider == "azure":
            return AsyncAzureOpenAI(
                api_key=config.api_key,
                azure_endpoint=config.azure_endpoint,
                api_version=config.azure_api_version,
                azure_ad_token=config.azure_ad_token,
            )

        return AsyncOpenAI(
            api_key=config.api_key,
            base_url=config.base_url,
            organization=config.organization,
        )

    @staticmethod
    def _build_messages(prompt: str, system_prompt: str | None) -> list[dict[str, str]]:
        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        return messages

    @staticmethod
    def _extract_text(content: Any) -> str:
        if content is None:
            return ""
        if isinstance(content, str):
            return content
        if isinstance(content, Iterable):
            chunks: list[str] = []
            for chunk in content:
                if isinstance(chunk, dict):
                    if chunk.get("type") == "text" and chunk.get("text"):
                        chunks.append(str(chunk["text"]))
                    elif chunk.get("type") == "tool_call":
                        chunks.append(str(chunk))
                else:
                    chunks.append(str(chunk))
            return "".join(chunks)
        return str(content)


class AzureBlobUploader:
    """Helper that uploads binary artifacts to Azure Blob Storage."""

    def __init__(self, config: AzureBlobStorageConfig) -> None:
        self._config = config
        self._service_client = BlobServiceClient.from_connection_string(
            config.connection_string
        )
        self._container_client = self._service_client.get_container_client(
            config.container
        )

    async def upload(
        self,
        *,
        data: bytes,
        category: str,
        extension: str,
        content_type: str,
    ) -> str:
        blob_name = self._build_blob_name(category, extension)
        blob_client = self._container_client.get_blob_client(blob_name)
        content_settings = ContentSettings(content_type=content_type)
        await blob_client.upload_blob(
            data,
            overwrite=True,
            content_settings=content_settings,
        )
        return f"{self._config.public_base_url}/{blob_name}"

    def _build_blob_name(self, category: str, extension: str) -> str:
        timestamp = datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%S")
        unique = uuid.uuid4().hex
        sanitized_category = self._root_for_category(category)
        filename = f"{timestamp}-{unique}.{extension}"
        segments = [
            segment
            for segment in (self._config.path_prefix, sanitized_category, filename)
            if segment
        ]
        return "/".join(segments)

    def _root_for_category(self, category: str) -> str:
        normalized = (category or "media").strip("/") or "media"
        mapping = {
            "images": self._config.image_root,
            "image": self._config.image_root,
            "audio": self._config.audio_root,
            "videos": self._config.video_root,
            "video": self._config.video_root,
        }
        return mapping.get(normalized, normalized)

    async def aclose(self) -> None:
        await self._service_client.close()

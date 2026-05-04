"""Qwen Vision LLM 实现。"""

from __future__ import annotations

import base64
import io
import json
import mimetypes
import socket
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from PIL import Image, UnidentifiedImageError

from libs.llm.base_vision_llm import BaseVisionLLM, VisionChatResponse


class QwenVisionLLMError(RuntimeError):
    """Qwen Vision LLM 调用错误。"""


class QwenVisionLLM(BaseVisionLLM):
    """基于 DashScope OpenAI-compatible 接口的 Qwen Vision 实现。"""

    provider_name = "qwen"
    default_base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    default_max_image_size = 2048

    def chat_with_image(
        self,
        text: str,
        image_path: str | bytes,
        trace: Any | None = None,
    ) -> VisionChatResponse:
        if not isinstance(text, str) or not text.strip():
            raise QwenVisionLLMError("qwen vision input error: text is required")

        image_url, image_metadata = self._prepare_image_url(image_path)
        payload = self._build_payload(text=text, image_url=image_url)
        request = Request(
            self._chat_endpoint(),
            data=json.dumps(payload).encode("utf-8"),
            headers=self._headers(),
            method="POST",
        )

        try:
            with urlopen(request, timeout=float(self.config.get("timeout", 30))) as response:
                data = json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            raise self._build_http_error(exc) from exc
        except URLError as exc:
            raise QwenVisionLLMError(f"qwen vision network error: {exc.reason}") from exc
        except (TimeoutError, socket.timeout) as exc:
            raise QwenVisionLLMError(f"qwen vision timeout error: {type(exc).__name__}") from exc
        except json.JSONDecodeError as exc:
            raise QwenVisionLLMError("qwen vision response error: invalid JSON") from exc

        content = self._parse_response_content(data)
        metadata = {
            "provider": self.provider_name,
            "model": self._model_name(),
            "image": image_metadata,
        }
        if trace is not None:
            metadata["trace"] = trace
        return VisionChatResponse(content=content, metadata=metadata)

    def preprocess_image(self, image_path: str | bytes) -> str:
        """将图片输入标准化为可直接发送给 Qwen 的 data URL。"""

        image_url, _ = self._prepare_image_url(image_path)
        return image_url

    def _prepare_image_url(self, image_input: str | bytes) -> tuple[str, dict[str, Any]]:
        image_bytes, mime_type, source = self._read_image_bytes(image_input)
        processed_bytes, dimensions = self._resize_image(image_bytes, mime_type)
        encoded = base64.b64encode(processed_bytes).decode("ascii")
        return (
            f"data:{mime_type};base64,{encoded}",
            {
                "source": source,
                "mime_type": mime_type,
                "original_bytes": len(image_bytes),
                "processed_bytes": len(processed_bytes),
                "original_size": {
                    "width": dimensions["original"][0],
                    "height": dimensions["original"][1],
                },
                "processed_size": {
                    "width": dimensions["processed"][0],
                    "height": dimensions["processed"][1],
                },
                "compressed": dimensions["original"] != dimensions["processed"],
            },
        )

    def _read_image_bytes(self, image_input: str | bytes) -> tuple[bytes, str, str]:
        if isinstance(image_input, str):
            if self._is_file_path(image_input):
                path = Path(image_input)
                mime_type = mimetypes.guess_type(path.name)[0] or "image/png"
                return path.read_bytes(), mime_type, "path"
            if image_input.startswith("data:image/"):
                return self._decode_data_url(image_input), self._mime_type_from_data_url(image_input), "data_url"
            return self._decode_base64_string(image_input), "image/png", "base64"

        if isinstance(image_input, bytes):
            try:
                decoded = self._decode_base64_string(image_input.decode("ascii"))
            except (UnicodeDecodeError, QwenVisionLLMError):
                return image_input, "image/png", "bytes"
            return decoded, "image/png", "base64"

        raise QwenVisionLLMError("qwen vision input error: image_path must be file path, base64 string, or bytes")

    def _resize_image(self, image_bytes: bytes, mime_type: str) -> tuple[bytes, dict[str, tuple[int, int]]]:
        max_image_size = int(self.config.get("max_image_size", self.default_max_image_size))
        if max_image_size <= 0:
            raise QwenVisionLLMError("qwen vision config error: max_image_size must be positive")

        try:
            with Image.open(io.BytesIO(image_bytes)) as image:
                original_size = image.size
                processed_image = image.copy()
                processed_image.thumbnail((max_image_size, max_image_size))
                processed_size = processed_image.size
                if processed_size == original_size:
                    return image_bytes, {"original": original_size, "processed": processed_size}

                output = io.BytesIO()
                image_format = self._resolve_image_format(mime_type, processed_image.mode)
                save_kwargs: dict[str, Any] = {}
                if image_format == "JPEG":
                    processed_image = processed_image.convert("RGB")
                    save_kwargs["quality"] = 90
                processed_image.save(output, format=image_format, **save_kwargs)
                return output.getvalue(), {"original": original_size, "processed": processed_size}
        except UnidentifiedImageError as exc:
            raise QwenVisionLLMError("qwen vision input error: unsupported image format") from exc

    def _build_payload(self, text: str, image_url: str) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self._model_name(),
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": text},
                        {"type": "image_url", "image_url": {"url": image_url}},
                    ],
                }
            ],
        }
        for key in ("temperature", "max_tokens", "top_p"):
            if key in self.config:
                payload[key] = self.config[key]
        return payload

    def _chat_endpoint(self) -> str:
        base_url = str(self.config.get("base_url", self.default_base_url)).rstrip("/")
        if not base_url:
            raise QwenVisionLLMError("qwen vision config error: base_url is required")
        return f"{base_url}/chat/completions"

    def _model_name(self) -> str:
        model = self.config.get("model")
        if not model:
            raise QwenVisionLLMError("qwen vision config error: model is required")
        return str(model)

    def _headers(self) -> dict[str, str]:
        api_key = self.config.get("api_key")
        if not api_key:
            raise QwenVisionLLMError("qwen vision config error: api_key is required")
        return {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}

    def _parse_response_content(self, data: dict[str, Any]) -> str:
        try:
            content = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise QwenVisionLLMError(
                "qwen vision response error: missing choices[0].message.content"
            ) from exc

        if isinstance(content, str):
            return content
        if isinstance(content, list):
            text_parts = [
                block.get("text", "")
                for block in content
                if isinstance(block, dict) and block.get("type") == "text"
            ]
            merged = "".join(text_parts).strip()
            if merged:
                return merged
        raise QwenVisionLLMError("qwen vision response error: content must be string or text blocks")

    def _build_http_error(self, exc: HTTPError) -> QwenVisionLLMError:
        error_code = ""
        error_message = ""
        try:
            payload = json.loads(exc.read().decode("utf-8"))
        except (OSError, ValueError, json.JSONDecodeError):
            payload = {}

        if isinstance(payload, dict):
            error = payload.get("error")
            if isinstance(error, dict):
                error_code = str(error.get("code", "")).strip()
                error_message = str(error.get("message", "")).strip()

        details = [f"qwen vision HTTP error: {exc.code}"]
        if error_code:
            details.append(f"code={error_code}")
        if error_message:
            details.append(error_message)
        return QwenVisionLLMError("; ".join(details))

    @staticmethod
    def _is_file_path(value: str) -> bool:
        if not value or "\n" in value or "\r" in value:
            return False
        return Path(value).exists()

    @staticmethod
    def _decode_data_url(value: str) -> bytes:
        _, _, payload = value.partition(",")
        if not payload:
            raise QwenVisionLLMError("qwen vision input error: invalid data URL")
        return QwenVisionLLM._decode_base64_string(payload)

    @staticmethod
    def _mime_type_from_data_url(value: str) -> str:
        header, _, _ = value.partition(",")
        mime_type = header.removeprefix("data:").split(";")[0].strip()
        return mime_type or "image/png"

    @staticmethod
    def _decode_base64_string(value: str) -> bytes:
        try:
            return base64.b64decode(value, validate=True)
        except (ValueError, TypeError) as exc:
            raise QwenVisionLLMError("qwen vision input error: invalid base64 image") from exc

    @staticmethod
    def _resolve_image_format(mime_type: str, mode: str) -> str:
        if mime_type == "image/jpeg":
            return "JPEG"
        if mime_type == "image/webp":
            return "WEBP"
        if mime_type == "image/gif":
            return "GIF"
        if mode == "RGBA":
            return "PNG"
        return "PNG"

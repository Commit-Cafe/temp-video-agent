import asyncio
import logging
import os
from typing import Any, Callable

logger = logging.getLogger("video-to-notes.transcriber")

_whisper_model: Any = None
_whisper_model_size: str | None = None


def _get_whisper_model(
    model_size: str, device: str, compute_type: str
):
    global _whisper_model, _whisper_model_size
    if _whisper_model is not None and _whisper_model_size == model_size:
        return _whisper_model

    from faster_whisper import WhisperModel

    logger.info(
        f"Loading faster-whisper model '{model_size}' on {device}/{compute_type}..."
    )
    _whisper_model = WhisperModel(
        model_size, device=device, compute_type=compute_type
    )
    _whisper_model_size = model_size
    logger.info("faster-whisper model loaded")
    return _whisper_model


async def _transcribe_faster_whisper(
    audio_path: str,
    model_size: str,
    device: str,
    compute_type: str,
    language: str | None,
    progress_callback: Callable[[float], None] | None = None,
) -> str:
    model = _get_whisper_model(model_size, device, compute_type)

    loop = asyncio.get_running_loop()

    def _run():
        segments, info = model.transcribe(
            audio_path,
            language=language,
            beam_size=5,
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 500},
        )

        detected_lang = info.language
        duration = info.duration or 0
        logger.info(
            f"Detected language: {detected_lang}, duration: {duration:.1f}s"
        )

        parts = []
        for segment in segments:
            parts.append(segment.text.strip())
            if progress_callback and duration > 0:
                progress_callback(min(segment.end / duration, 1.0))

        return "\n".join(parts)

    return await loop.run_in_executor(None, _run)


async def _transcribe_whisper_api(
    audio_path: str,
    api_key: str,
    language: str | None,
) -> str:
    try:
        from openai import AsyncOpenAI
    except ImportError:
        raise RuntimeError("openai package is not installed. Run: pip install openai")

    client = AsyncOpenAI(api_key=api_key)
    with open(audio_path, "rb") as f:
        response = await client.audio.transcriptions.create(
            model="whisper-1",
            file=f,
            language=language,
            response_format="text",
        )
    return response


async def transcribe(
    audio_path: str,
    model_size: str = "small",
    device: str = "cpu",
    compute_type: str = "int8",
    language: str | None = None,
    progress_callback: Callable[[float], None] | None = None,
    whisper_api_key: str = "",
) -> str:
    """Transcribe audio to text. Uses faster-whisper with Whisper API fallback."""

    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    # Primary path: faster-whisper
    try:
        return await _transcribe_faster_whisper(
            audio_path=audio_path,
            model_size=model_size,
            device=device,
            compute_type=compute_type,
            language=language,
            progress_callback=progress_callback,
        )
    except ImportError:
        logger.warning("faster-whisper not available, trying Whisper API fallback")
    except Exception as e:
        logger.warning(f"faster-whisper failed: {e}, trying Whisper API fallback")

    # Fallback path: Whisper API
    if not whisper_api_key:
        raise RuntimeError(
            "faster-whisper is not available and no WHISPER_API_KEY is configured. "
            "Install faster-whisper (pip install faster-whisper) or set WHISPER_API_KEY in .env."
        )

    logger.info("Using Whisper API for transcription")
    return await _transcribe_whisper_api(
        audio_path=audio_path,
        api_key=whisper_api_key,
        language=language,
    )

"""Tests for the video-to-notes pipeline components."""

import os
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from pipeline.downloader import download_audio
from pipeline.structurer import _split_transcript, _estimate_tokens, structure_notes
from pipeline.transcriber import transcribe

# --- Config ---


@patch.dict(
    os.environ,
    {
        "DEEPSEEK_API_KEY": "test-key",
        "DEEPSEEK_BASE_URL": "https://api.deepseek.com/v1",
    },
)
def test_config_loading():
    from config import Config

    c = Config.from_env()
    assert c.deepseek.api_key == "test-key"
    assert c.deepseek.base_url == "https://api.deepseek.com/v1"
    assert c.deepseek.model_name == "deepseek-v4-pro"
    assert c.default_model == "deepseek-v4-pro"


def test_config_model_routing():
    from config import Config

    c = Config.from_env()
    assert c.get_llm_config("minimax-m2.7") == c.minimax
    assert c.get_llm_config("glm-5.1") == c.zhipuai
    assert c.get_llm_config("deepseek-v4-pro") == c.deepseek
    assert c.get_llm_config("unknown-model") == c.deepseek  # fallback


# --- Downloader ---


def test_downloader_empty_url():
    import pytest

    with pytest.raises(ValueError, match="URL is required"):
        import asyncio

        asyncio.run(download_audio("", "/tmp"))


def test_downloader_invalid_url():
    import pytest

    with pytest.raises(Exception):
        import asyncio

        asyncio.run(download_audio("not-a-valid-url-$$$", "/tmp"))


# --- Transcriber ---


def test_transcriber_file_not_found():
    import pytest

    with pytest.raises(FileNotFoundError):
        import asyncio

        asyncio.run(transcribe("/nonexistent/path/audio.mp3"))


# --- Structurer ---


def test_split_transcript_short():
    text = "Short transcript under the chunk limit."
    chunks = _split_transcript(text)
    assert len(chunks) == 1
    assert chunks[0] == text


def test_split_transcript_long():
    # Create text exceeding CHUNK_SIZE
    para = "A" * 17000
    chunks = _split_transcript(para)
    assert len(chunks) >= 2


def test_estimate_tokens():
    assert _estimate_tokens("hello world") > 0
    assert _estimate_tokens("你好世界") > 0
    assert _estimate_tokens("") == 0


@pytest.mark.asyncio
async def test_structure_notes_empty_transcript():
    from config import Config

    c = Config.from_env()
    with pytest.raises(ValueError, match="empty"):
        await structure_notes("", "deepseek-v4-pro", c.deepseek)


@pytest.mark.asyncio
async def test_structure_notes_success():
    from config import Config, LLMConfig

    with patch(
        "pipeline.structurer._call_llm",
        new_callable=AsyncMock,
        return_value="## Test Notes\n\n### TL;DR\n- Point 1\n- Point 2",
    ):
        llm_config = LLMConfig(
            api_key="test-key",
            base_url="https://api.deepseek.com/v1",
            model_name="deepseek-v4-pro",
        )
        result = await structure_notes(
            "This is a test transcript about machine learning.",
            "deepseek-v4-pro",
            llm_config,
        )
        assert "## Test Notes" in result
        assert "Point 1" in result


@pytest.mark.asyncio
async def test_structure_notes_missing_api_key():
    from config import LLMConfig

    llm_config = LLMConfig(
        api_key="",
        base_url="https://api.deepseek.com/v1",
        model_name="deepseek-v4-pro",
    )
    with pytest.raises(ValueError, match="API key not configured"):
        await structure_notes(
            "Test transcript",
            "deepseek-v4-pro",
            llm_config,
        )

"""
Video-to-Notes MCP Server.

Converts video URLs into structured Markdown notes.
Pipeline: yt-dlp download → faster-whisper ASR → LLM structuring.

Usage:
    python server.py    # starts stdio MCP server
"""

import logging
import os
import sys
from pathlib import Path

# Ensure project root is the working directory so .env loads correctly
os.chdir(Path(__file__).resolve().parent)

from mcp.server.fastmcp import FastMCP

from config import Config
from pipeline.downloader import download_audio
from pipeline.structurer import structure_notes
from pipeline.transcriber import transcribe

logging.basicConfig(
    level=logging.INFO,
    format="[video-to-notes] %(levelname)s: %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger("video-to-notes")

config = Config.from_env()

mcp = FastMCP(
    name="video-to-notes",
    instructions="""
You are a video-to-notes assistant. Convert video URLs into structured markdown notes.

Available tool: `video_to_notes`
- Takes a video URL (YouTube, Bilibili, etc.)
- Optional: model (minimax-m2.7, glm-5.1, deepseek-v4-pro), language (auto/zh/en), output_style (detailed/summary/bullet_points)
- Pipeline: download audio → speech-to-text → LLM structuring
- Returns markdown notes

Workflow:
1. Call `video_to_notes` with the video URL
2. Wait for the pipeline to complete (may take a few minutes for long videos)
3. Present or save the returned markdown notes to the user
""".strip(),
)


@mcp.tool()
async def video_to_notes(
    url: str,
    model: str = "deepseek-v4-pro",
    language: str = "auto",
    output_style: str = "detailed",
) -> str:
    """Convert a video URL into structured markdown notes.

    Args:
        url: Video URL (YouTube, Bilibili, or any yt-dlp supported site)
        model: LLM for structuring notes. Options: minimax-m2.7, glm-5.1, deepseek-v4-pro
        language: Language hint for transcription. Options: auto, zh, en
        output_style: Note detail level. Options: detailed, summary, bullet_points

    Returns:
        Structured markdown notes.
    """
    # --- Validate inputs ---
    if not url or not url.strip():
        return "Error: Please provide a video URL."

    valid_models = {"minimax-m2.7", "glm-5.1", "deepseek-v4-pro"}
    if model not in valid_models:
        return f"Error: Unknown model '{model}'. Available: {', '.join(sorted(valid_models))}"

    valid_languages = {"auto", "zh", "en"}
    if language not in valid_languages:
        return f"Error: Unknown language '{language}'. Use: auto, zh, or en"

    valid_styles = {"detailed", "summary", "bullet_points"}
    if output_style not in valid_styles:
        return f"Error: Unknown output_style '{output_style}'. Use: detailed, summary, or bullet_points"

    # --- Get LLM config ---
    llm_config = config.get_llm_config(model)
    if not llm_config.api_key:
        env_map = {
            "minimax-m2.7": "MINIMAX_API_KEY",
            "glm-5.1": "ZHIPUAI_API_KEY",
            "deepseek-v4-pro": "DEEPSEEK_API_KEY",
        }
        return (
            f"Error: API key not configured for {model}. "
            f"Set {env_map.get(model, 'the API key')} in the .env file."
        )

    # --- Pipeline ---
    temp_dir = Path(config.temp_dir)
    temp_dir.mkdir(parents=True, exist_ok=True)

    audio_path = None
    try:
        # Stage 1: Download audio
        logger.info(f"Downloading audio from: {url}")

        def download_progress(pct: float):
            logger.info(f"Download progress: {pct:.0%}")

        audio_path = await download_audio(
            url, str(temp_dir), progress_callback=download_progress
        )
        logger.info(f"Audio extracted: {audio_path}")

        # Stage 2: Transcribe
        logger.info("Transcribing audio to text...")

        def transcribe_progress(pct: float):
            logger.info(f"Transcription progress: {pct:.0%}")

        whisper_lang = None if language == "auto" else language
        transcript = await transcribe(
            audio_path=audio_path,
            model_size=config.whisper_model_size,
            device=config.whisper_device,
            compute_type=config.whisper_compute_type,
            language=whisper_lang,
            progress_callback=transcribe_progress,
            whisper_api_key=config.whisper_api_key,
        )

        if not transcript or not transcript.strip():
            return (
                "Error: No speech detected in the video. "
                "The video may have no audio track or the speech is inaudible."
            )

        logger.info(
            f"Transcription complete: {len(transcript)} chars"
        )

        # Stage 3: Structure notes
        logger.info(f"Structuring notes with {model}...")
        notes = await structure_notes(
            transcript=transcript,
            model=model,
            llm_config=llm_config,
            language=language,
            output_style=output_style,
        )

        logger.info("Notes generated successfully")
        return notes

    except ValueError as e:
        return f"Input error: {e}"
    except FileNotFoundError as e:
        return f"File error: {e}"
    except RuntimeError as e:
        return f"Processing error: {e}"
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        return f"Unexpected error processing video: {e}"

    finally:
        # Cleanup temp audio
        if audio_path and os.path.exists(audio_path):
            try:
                os.remove(audio_path)
                logger.info("Cleaned up temp audio file")
            except OSError:
                pass


def main():
    logger.info(
        f"Starting video-to-notes MCP server "
        f"(default model: {config.default_model}, "
        f"whisper: {config.whisper_model_size}/{config.whisper_device})"
    )
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()

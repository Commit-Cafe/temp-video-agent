import asyncio
import logging
import os
import sys
from pathlib import Path
from typing import Callable

logger = logging.getLogger("video-to-notes.downloader")


def _sync_download(
    url: str,
    output_dir: str,
    progress_callback: Callable[[float], None] | None = None,
) -> str:
    """Synchronous yt-dlp download. Runs in a thread pool executor."""

    def _progress_hook(d: dict):
        if progress_callback and d["status"] == "downloading":
            total = d.get("total_bytes") or d.get("total_bytes_estimate")
            downloaded = d.get("downloaded_bytes", 0)
            if total:
                progress_callback(min(downloaded / total, 1.0))

    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": f"{output_dir}/%(id)s.%(ext)s",
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "mp3",
                "preferredquality": "192",
            }
        ],
        "progress_hooks": [_progress_hook],
        "quiet": True,
        "no_warnings": True,
        "extract_flat": False,
    }

    try:
        from yt_dlp import YoutubeDL

        with YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            video_id = info["id"]
            audio_path = os.path.join(output_dir, f"{video_id}.mp3")
            if not os.path.exists(audio_path):
                raise FileNotFoundError(
                    f"Audio file not found after download: {audio_path}"
                )
            logger.info(f"Downloaded audio: {audio_path}")
            return audio_path

    except ImportError:
        raise RuntimeError(
            "yt-dlp is not installed. Run: pip install yt-dlp"
        )
    except Exception as e:
        error_msg = str(e)
        if "Unsupported URL" in error_msg or "not a valid URL" in error_msg.lower():
            raise ValueError(f"Unsupported URL: {url}")
        if "ffmpeg" in error_msg.lower() or "ffprobe" in error_msg.lower():
            raise RuntimeError(
                "ffmpeg/ffprobe not found. Install ffmpeg: https://ffmpeg.org/download.html"
            )
        if "HTTP Error 403" in error_msg or "HTTP Error 404" in error_msg:
            raise RuntimeError(
                f"Video is not accessible (403/404): {url}\n"
                "The video may be private, geo-restricted, or deleted."
            )
        if "This video is unavailable" in error_msg:
            raise RuntimeError(
                f"Video is unavailable: {url}\n"
                "The video may be age-restricted or region-locked."
            )
        raise RuntimeError(f"Failed to download video: {error_msg}")


async def download_audio(
    url: str,
    output_dir: str,
    progress_callback: Callable[[float], None] | None = None,
) -> str:
    if not url or not url.strip():
        raise ValueError("URL is required")

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(
        None, _sync_download, url, output_dir, progress_callback
    )

# Video-to-Notes MCP Server

Convert any video URL into structured, high-quality Markdown notes. An MCP (Model Context Protocol) server designed for use with Claude Code and other MCP-compatible agents.

## Pipeline

```
Video URL → yt-dlp (audio) → faster-whisper (ASR) → LLM (structuring) → Markdown notes
```

## Supported Platforms

YouTube, Bilibili, and [any site yt-dlp supports](https://github.com/yt-dlp/yt-dlp/blob/master/supportedsites.md).

## Supported LLMs

| Model | Provider | Env Key |
|-------|----------|---------|
| `deepseek-v4-pro` | DeepSeek | `DEEPSEEK_API_KEY` |
| `glm-5.1` | ZhipuAI | `ZHIPUAI_API_KEY` |
| `minimax-m2.7` | MiniMax | `MINIMAX_API_KEY` |

All three use OpenAI-compatible APIs.

## Prerequisites

- Python 3.10+
- ffmpeg (required by yt-dlp for audio extraction)
- At least one LLM API key

### Install ffmpeg

- **Windows**: `winget install ffmpeg` or download from [ffmpeg.org](https://ffmpeg.org/download.html)
- **macOS**: `brew install ffmpeg`
- **Linux**: `apt install ffmpeg` / `dnf install ffmpeg`

## Setup

```bash
# 1. Clone
git clone https://github.com/Commit-Cafe/temp-video-agent.git
cd temp-video-agent

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure API keys
cp .env.example .env
# Edit .env — fill in at least one LLM API key

# 4. (Optional) Download Whisper model ahead of time
python -c "from faster_whisper import WhisperModel; WhisperModel('small')"
```

## MCP Registration

Add to your project's `.mcp.json`:

```json
{
  "mcpServers": {
    "video-to-notes": {
      "command": "python",
      "args": ["server.py"],
      "cwd": "<path-to>/temp-video-agent"
    }
  }
}
```

## Tool: `video_to_notes`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `url` | string | (required) | Video URL |
| `model` | string | `deepseek-v4-pro` | LLM for note structuring |
| `language` | string | `auto` | Transcription language: `auto`, `zh`, `en` |
| `output_style` | string | `detailed` | Note detail: `detailed`, `summary`, `bullet_points` |

### Example Usage (via Claude Code)

> "Take notes on https://www.youtube.com/watch?v=xxxxx using glm-5.1 in summary mode"

Claude Code will call:

```
video_to_notes(url="https://www.youtube.com/watch?v=xxxxx", model="glm-5.1", output_style="summary")
```

### Note Output Format

```markdown
## Video Title

### TL;DR
- Key takeaway 1
- Key takeaway 2
- Key takeaway 3

### Section 1
- Supporting detail
- Example

### Section 2
**Key concept** explained in detail...

> Notable quote [05:23]

### Summary & Next Steps
- Action item
```

## Configuration Reference

| Variable | Default | Description |
|----------|---------|-------------|
| `DEEPSEEK_API_KEY` | — | DeepSeek API key |
| `ZHIPUAI_API_KEY` | — | ZhipuAI (GLM) API key |
| `MINIMAX_API_KEY` | — | MiniMax API key |
| `DEFAULT_MODEL` | `deepseek-v4-pro` | Default LLM |
| `WHISPER_MODEL_SIZE` | `small` | Whisper model: `tiny`, `base`, `small`, `medium`, `large-v3` |
| `WHISPER_DEVICE` | `cpu` | Inference device: `cpu`, `cuda` |
| `WHISPER_API_KEY` | — | OpenAI API key for Whisper API fallback (optional) |
| `TEMP_DIR` | `./temp_audio` | Temp audio file directory |

## Running Tests

```bash
pip install pytest pytest-asyncio
python -m pytest tests/ -v
```

## Known Limitations

- Videos without an audio track (silent footage) will fail at the transcription stage.
- Very long videos (>2 hours) may require significant RAM for Whisper transcription.
- Geo-restricted or private videos cannot be downloaded.
- The first run downloads the Whisper model (~500MB for `small`), which takes time.

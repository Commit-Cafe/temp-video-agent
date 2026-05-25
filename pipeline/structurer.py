import asyncio
import logging
from typing import List

from config import LLMConfig

logger = logging.getLogger("video-to-notes.structurer")

# Approximate: 1 token ≈ 2 characters for Chinese, 4 characters for English
# Target ~8000 tokens per chunk for safe margins
CHUNK_SIZE = 16000  # characters
CHUNK_OVERLAP = 2000  # characters

SYSTEM_PROMPT = """You are a professional note-taking assistant. Convert raw video transcripts into well-structured markdown notes.

## Guidelines
1. Identify the main topic and create a clear title (use `##` for title)
2. Start with a **TL;DR** section: 3-5 key takeaways in bullet points
3. Extract and organize content with hierarchical headings (`###` for sections, `####` for sub-sections)
4. Use bullet points (`-`) for supporting details and examples
5. Use `>` blockquotes for notable quotes or important statements
6. **Bold** key concepts, terminology, and important names
7. Preserve timestamps in `[MM:SS]` or `[HH:MM:SS]` format when available
8. Remove filler words (um, uh, "you know", "就是说", "然后呢") while preserving meaning
9. Complete incomplete sentences silently — do not add information not present in the transcript
10. Keep the output language consistent with the transcript language
11. End with a **Summary & Action Items** section when applicable
12. NEVER fabricate facts, data, or quotes not present in the transcript

## Output Format
```
## [Video Title / Main Topic]

### TL;DR
- Key point 1
- Key point 2
- ...

### [Section 1 Title]
- Detail point
- ...

#### [Sub-section if needed]

### [Section 2 Title]
...

### Key Quotes
> Notable quote [MM:SS]

### Summary & Next Steps (if applicable)
```
"""

CHUNK_PROMPT = """You are processing PART {part_num} of {total_parts} of a video transcript.
Extract and structure the key points from this section into markdown notes.
Keep the original section headings and timestamps.
Do NOT add a TL;DR or summary — only the merging step will create those.

Transcript part {part_num}:
{chunk_text}"""

MERGE_PROMPT = """You are merging {total_parts} structured note sections from a single video into ONE cohesive document.

Combine the sections below into a final, unified markdown note. Follow these rules:
1. Create a single `## Title` for the whole video
2. Write a `### TL;DR` covering all parts (3-5 bullet points)
3. Merge overlapping sections — if multiple parts discuss the same topic, consolidate
4. Keep all timestamps intact
5. Preserve all key quotes
6. Add a final `### Summary & Next Steps` section

Sections to merge:

{combined_sections}"""


def _estimate_tokens(text: str) -> int:
    """Rough token estimation: ~2 chars per token for mixed Chinese/English text."""
    return len(text) // 2


def _split_transcript(transcript: str) -> List[str]:
    """Split transcript into overlapping chunks by paragraph boundaries."""
    paragraphs = transcript.split("\n\n")
    chunks = []
    current = ""
    for para in paragraphs:
        if len(current) + len(para) > CHUNK_SIZE and current:
            chunks.append(current.strip())
            overlap = current[-CHUNK_OVERLAP:] if len(current) > CHUNK_OVERLAP else ""
            current = overlap + "\n\n" + para if overlap else para
        else:
            current = current + "\n\n" + para if current else para

    if current.strip():
        # If a single element exceeds CHUNK_SIZE, brute-force split it
        if len(current) > CHUNK_SIZE:
            for i in range(0, len(current), CHUNK_SIZE - CHUNK_OVERLAP):
                chunk = current[i : i + CHUNK_SIZE]
                if chunk.strip():
                    chunks.append(chunk.strip())
        else:
            chunks.append(current.strip())
    return chunks


async def _call_llm(
    llm_config: LLMConfig,
    messages: list,
    temperature: float = 0.3,
    max_tokens: int = 8192,
    retries: int = 3,
) -> str:
    """Call LLM with retry logic for rate limiting."""
    try:
        from openai import AsyncOpenAI
    except ImportError:
        raise RuntimeError("openai package is not installed. Run: pip install openai")

    if not llm_config.api_key:
        raise ValueError(
            f"API key not configured for model '{llm_config.model_name}'. "
            "Set the corresponding API key in .env file."
        )

    client = AsyncOpenAI(
        api_key=llm_config.api_key,
        base_url=llm_config.base_url,
        timeout=120.0,
    )

    for attempt in range(retries):
        try:
            logger.info(
                f"Calling {llm_config.model_name} "
                f"(attempt {attempt + 1}/{retries})..."
            )
            response = await client.chat.completions.create(
                model=llm_config.model_name,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            content = response.choices[0].message.content
            if not content:
                raise RuntimeError("LLM returned empty response")
            return content

        except Exception as e:
            error_text = str(e).lower()
            is_rate_limit = any(
                kw in error_text for kw in ["429", "rate limit", "too many requests"]
            )
            is_last = attempt == retries - 1

            if is_rate_limit and not is_last:
                wait = 2 ** (attempt + 1)
                logger.warning(f"Rate limited, retrying in {wait}s...")
                await asyncio.sleep(wait)
            elif is_last:
                raise RuntimeError(
                    f"LLM call failed after {retries} attempts: {e}"
                )
            else:
                raise RuntimeError(f"LLM call failed: {e}")


async def _structure_single_chunk(
    transcript: str, llm_config: LLMConfig, output_style: str
) -> str:
    """Structure a single transcript chunk with full formatting."""
    style_instructions = {
        "detailed": "Provide comprehensive notes with detailed explanations.",
        "summary": "Keep notes concise — focus on main points only, no long explanations.",
        "bullet_points": "Use primarily bullet-point format. Minimize prose paragraphs.",
    }

    user_prompt = f"""Here is the transcript. Output style: {style_instructions.get(output_style, style_instructions['detailed'])}

Transcript:
{transcript}"""

    return await _call_llm(
        llm_config=llm_config,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
    )


async def _structure_with_chunking(
    transcript: str, llm_config: LLMConfig
) -> str:
    """Process long transcript via chunking + merging."""
    chunks = _split_transcript(transcript)
    total = len(chunks)
    logger.info(f"Transcript split into {total} chunks for processing")

    # Process each chunk
    chunk_results = []
    for i, chunk in enumerate(chunks):
        logger.info(f"Processing chunk {i + 1}/{total}...")
        result = await _call_llm(
            llm_config=llm_config,
            messages=[
                {
                    "role": "system",
                    "content": "You process video transcript chunks into structured notes.",
                },
                {
                    "role": "user",
                    "content": CHUNK_PROMPT.format(
                        part_num=i + 1,
                        total_parts=total,
                        chunk_text=chunk,
                    ),
                },
            ],
        )
        chunk_results.append(result)

    # Merge chunks into final notes
    if total == 1:
        return chunk_results[0]

    logger.info(f"Merging {total} chunk results...")
    combined = "\n\n---\n\n".join(
        f"## Part {i + 1}\n{r}" for i, r in enumerate(chunk_results)
    )
    return await _call_llm(
        llm_config=llm_config,
        messages=[
            {
                "role": "system",
                "content": "You merge multiple note sections into one cohesive markdown document.",
            },
            {
                "role": "user",
                "content": MERGE_PROMPT.format(
                    total_parts=total,
                    combined_sections=combined,
                ),
            },
        ],
        max_tokens=16384,
    )


async def structure_notes(
    transcript: str,
    model: str,
    llm_config: LLMConfig,
    language: str = "auto",
    output_style: str = "detailed",
) -> str:
    """Convert raw transcript into structured markdown notes using LLM."""

    if not transcript or not transcript.strip():
        raise ValueError("Transcript is empty — nothing to structure")

    estimated_tokens = _estimate_tokens(transcript)
    logger.info(
        f"Transcript length: {len(transcript)} chars, "
        f"~{estimated_tokens} tokens est."
    )

    # For short transcripts, process in one shot
    if len(transcript) <= CHUNK_SIZE:
        return await _structure_single_chunk(
            transcript=transcript,
            llm_config=llm_config,
            output_style=output_style,
        )

    # For long transcripts, chunk and merge
    logger.info("Transcript too long, using chunked processing")
    return await _structure_with_chunking(
        transcript=transcript,
        llm_config=llm_config,
    )

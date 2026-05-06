# video-analyzer-ai

Extract transcripts and analyze videos with AI. YouTube, Twitter/X, TikTok, Instagram, and [1000+ sites](https://github.com/yt-dlp/yt-dlp/blob/master/supportedsites.md) supported.

```bash
va "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
va "https://x.com/user/status/123456" -a
va "https://www.tiktok.com/@user/video/123456" -a -v
```

## How it works

```
URL → YouTube captions (free) → transcript
        ↓ (no captions?)
      yt-dlp audio download → OpenRouter transcription → transcript
        ↓ (--analyze?)
      LLM analysis via any OpenAI-compatible API → structured breakdown
        ↓ (--verify?)
      Second pass with prompt cache optimization → verified result
```

- **YouTube**: tries built-in captions first (free, instant), falls back to audio transcription
- **Everything else**: downloads audio via yt-dlp, transcribes via OpenRouter
- **Analysis**: sends transcript to any OpenAI-compatible endpoint (OpenRouter, OpenAI, Gemini, local models)
- **Verification**: double-pass analysis where the second call reuses the prompt cache (~66% cheaper)
- **Caching**: transcripts, metadata, and analysis results cached locally with human-readable filenames

## Install

Requires Python 3.13+ and [yt-dlp](https://github.com/yt-dlp/yt-dlp).

```bash
# Install yt-dlp if you don't have it
pip install yt-dlp

# Install video-analyzer-ai
git clone https://github.com/testy-cool/video-analyzer-ai.git
cd video-analyzer-ai
cp .env.example .env  # add your API keys
pip install -e .
```

Or with [uv](https://docs.astral.sh/uv/):

```bash
uv tool install --editable .
```

The CLI command is `va`.

## Configuration

Copy `.env.example` to `.env` and fill in your keys:

```bash
# Required for audio transcription fallback (when captions unavailable)
OPENROUTER_API_KEY=sk-or-v1-...

# Analysis LLM — any OpenAI-compatible endpoint
ANALYSIS_BASE_URL=https://openrouter.ai/api/v1/chat/completions
ANALYSIS_API_KEY=sk-or-v1-...
ANALYSIS_MODEL=openai/gpt-4o-mini

# Optional: Langfuse tracing
LANGFUSE_SECRET_KEY=
LANGFUSE_PUBLIC_KEY=
LANGFUSE_BASE_URL=

# Optional: Evomi residential proxy (for rate-limited sources)
EVOMI_USER=
EVOMI_PASS=
```

**Zero-config mode**: YouTube captions work without any API keys. You only need `OPENROUTER_API_KEY` for videos without captions, and `ANALYSIS_API_KEY` for the `-a` flag.

## Usage

```bash
# Transcript only (free for YouTube with captions)
va "https://www.youtube.com/watch?v=..."

# With timestamps
va "https://www.youtube.com/watch?v=..." -t

# AI analysis
va "https://www.youtube.com/watch?v=..." -a

# Analysis + verification (double-pass, uses prompt cache)
va "https://www.youtube.com/watch?v=..." -a -v

# Custom prompt
va "https://www.youtube.com/watch?v=..." -q "list every product mentioned"

# Custom prompt + verification
va "https://www.youtube.com/watch?v=..." -q "extract all claims" -v

# JSON output (pipe-friendly)
va "https://www.youtube.com/watch?v=..." -a -j | jq .analysis.summary

# Non-YouTube (requires OPENROUTER_API_KEY)
va "https://x.com/user/status/123456789"
va "https://www.tiktok.com/@user/video/123456789"

# Force audio transcription (skip captions)
va "https://www.youtube.com/watch?v=..." -f

# Different language
va "https://www.youtube.com/watch?v=..." -l ro

# Open cache folder
va -o

# Skip cache
va "https://www.youtube.com/watch?v=..." --no-cache
```

## Flags

| Flag | Short | Description |
|------|-------|-------------|
| `--analyze` | `-a` | Analyze transcript with LLM |
| `--verify` | `-v` | Double-pass verification (implies `-a`) |
| `--prompt` | `-q` | Custom analysis prompt (implies `-a`) |
| `--json-output` | `-j` | JSON output to stdout |
| `--timestamps` | `-t` | Show timestamps in text output |
| `--force-transcribe` | `-f` | Skip captions, use audio transcription |
| `--lang` | `-l` | Caption language codes (default: en) |
| `--no-metadata` | `-M` | Skip metadata fetching |
| `--proxy` | `-p` | Use Evomi residential proxy |
| `--model` | | Transcription model (default: `openai/gpt-4o-mini-transcribe`) |
| `--analysis-model` | | Analysis model (default: from env) |
| `--no-cache` | | Bypass all caching |
| `--open-cache` | `-o` | Open cache folder in file manager |

## Cache

Transcripts, metadata, and analysis results are cached at `~/.cache/video-analyzer/` with human-readable filenames:

```
AI-Engineer_MCP-UI-Extending-the-frontier_o-zkvb0iFDQ.transcript.json
AI-Engineer_MCP-UI-Extending-the-frontier_o-zkvb0iFDQ.analysis.gemini_gemini-3.1-flash-lite-preview.json
AI-Engineer_MCP-UI-Extending-the-frontier_o-zkvb0iFDQ.metadata.json
```

Analysis is cached per model — switching `--analysis-model` triggers a fresh analysis. Use `--no-cache` to bypass, or `va -o` to browse cached files.

## Cost

| Operation | Cost |
|-----------|------|
| YouTube captions | Free |
| Audio transcription (OpenRouter) | ~$0.0005/min |
| Analysis (depends on model) | ~$0.001-0.01/run |
| Verification pass | ~33% of analysis (prompt cache) |
| Cached results | Free |

## License

MIT

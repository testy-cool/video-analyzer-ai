#!/usr/bin/env python3
"""Video analyzer CLI — extract transcripts and analyze videos."""

import base64

import json
import os
import re
import subprocess
import sys
import tempfile

import click
import requests
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from rich import box

load_dotenv()
from langfuse import Langfuse
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api.proxies import WebshareProxyConfig

console = Console(stderr=True)
out = Console()

OPENROUTER_URL = "https://openrouter.ai/api/v1/audio/transcriptions"
DEFAULT_TRANSCRIPTION_MODEL = "openai/gpt-4o-mini-transcribe"

ANALYSIS_BASE_URL = os.environ.get("ANALYSIS_BASE_URL", "https://openrouter.ai/api/v1/chat/completions")
ANALYSIS_API_KEY = os.environ.get("ANALYSIS_API_KEY", os.environ.get("OPENROUTER_API_KEY", ""))
DEFAULT_ANALYSIS_MODEL = os.environ.get("ANALYSIS_MODEL", "openai/gpt-4o-mini")

CACHE_DIR = os.path.expanduser("~/.cache/video-analyzer")


def slugify(text: str, max_len: int = 60) -> str:
    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"[\s_]+", "-", text).strip("-")
    return text[:max_len].rstrip("-")


def cache_prefix(video_id: str) -> str:
    """Find existing cache files for this video ID regardless of slug."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    for f in os.listdir(CACHE_DIR):
        if video_id in f:
            prefix = f.split(".transcript.")[0].split(".analysis.")[0].split(".metadata.")[0]
            return prefix
    return video_id


def cache_set_prefix(video_id: str, meta: dict | None) -> str:
    """Build prefix with slug from metadata if available."""
    if meta and meta.get("author") and meta.get("title"):
        author_slug = slugify(meta["author"], 20)
        title_slug = slugify(meta["title"], 50)
        return f"{author_slug}_{title_slug}_{video_id}"
    return video_id


def cache_path(video_id: str, kind: str, prefix: str | None = None) -> str:
    os.makedirs(CACHE_DIR, exist_ok=True)
    p = prefix or cache_prefix(video_id)
    return os.path.join(CACHE_DIR, f"{p}.{kind}.json")


def cache_get(video_id: str, kind: str) -> dict | None:
    path = cache_path(video_id, kind)
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


def cache_set(video_id: str, kind: str, data: dict, meta: dict | None = None):
    prefix = cache_set_prefix(video_id, meta) if meta else None
    path = cache_path(video_id, kind, prefix=prefix)
    with open(path, "w") as f:
        json.dump(data, f, ensure_ascii=False)


ANALYSIS_PROMPT = """You are a video content analyst. Given a video transcript (and optional metadata), produce a structured analysis.

Be direct and concise. No filler.

Output JSON with these fields:
- summary: 2-3 sentence TL;DW
- key_points: array of the main takeaways (max 10)
- topics: array of topic tags
- sentiment: overall tone (positive/negative/neutral/mixed)
- notable_quotes: array of standout quotes from the transcript (max 5, exact words)
- action_items: array of actionable advice if any (empty array if none)"""


def extract_video_id(url_or_id: str) -> str:
    patterns = [
        r"(?:youtube\.com/watch\?v=|youtu\.be/|youtube\.com/embed/|youtube\.com/shorts/)([a-zA-Z0-9_-]{11})",
        r"^([a-zA-Z0-9_-]{11})$",
    ]
    for pattern in patterns:
        match = re.search(pattern, url_or_id)
        if match:
            return match.group(1)
    raise click.BadParameter(f"Could not extract video ID from: {url_or_id}")


def get_proxy_config():
    user = os.environ.get("EVOMI_USER")
    password = os.environ.get("EVOMI_PASS")
    if not user or not password:
        return None
    return WebshareProxyConfig(
        proxy_url=f"http://{user}:{password}@core-residential.evomi.com:1000"
    )


def fetch_metadata(video_id: str) -> dict | None:
    try:
        resp = requests.get(
            f"https://www.youtube.com/oembed?url=https://www.youtube.com/watch?v={video_id}&format=json",
            timeout=10,
        )
        if resp.status_code == 200:
            data = resp.json()
            return {
                "title": data.get("title"),
                "author": data.get("author_name"),
                "author_url": data.get("author_url"),
                "thumbnail": data.get("thumbnail_url"),
            }
    except Exception:
        pass
    return None


def fetch_transcript_captions(video_id: str, languages: list[str] | None = None) -> dict:
    proxy_config = get_proxy_config()

    kwargs = {}
    if proxy_config:
        kwargs["proxies"] = proxy_config

    ytt = YouTubeTranscriptApi(**kwargs)
    langs = languages or ["en", "en-US", "en-GB"]
    transcript = ytt.fetch(video_id, languages=langs)

    segments = []
    full_text_parts = []
    for snippet in transcript:
        segments.append({
            "start": round(snippet.start, 1),
            "duration": round(snippet.duration, 1),
            "text": snippet.text,
        })
        full_text_parts.append(snippet.text)

    return {
        "segments": segments,
        "full_text": " ".join(full_text_parts),
        "language": langs[0],
        "source": "youtube_captions",
    }


def download_audio(video_id: str, tmpdir: str) -> str:
    out_path = os.path.join(tmpdir, "audio.%(ext)s")
    cmd = [
        "yt-dlp", "-x",
        "--audio-format", "mp3",
        "--audio-quality", "9",
        "--max-filesize", "24M",
        "-o", out_path,
        f"https://www.youtube.com/watch?v={video_id}",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"yt-dlp failed: {result.stderr.strip()}")

    mp3_path = os.path.join(tmpdir, "audio.mp3")
    if not os.path.exists(mp3_path):
        raise RuntimeError("yt-dlp did not produce an mp3 file")
    return mp3_path


def transcribe_audio(audio_path: str, api_key: str, model: str = DEFAULT_TRANSCRIPTION_MODEL) -> dict:
    with open(audio_path, "rb") as f:
        audio_b64 = base64.standard_b64encode(f.read()).decode()

    file_size_mb = os.path.getsize(audio_path) / (1024 * 1024)
    console.log(f"Transcribing [bold]{file_size_mb:.1f}MB[/bold] audio with [cyan]{model}[/cyan]")

    resp = requests.post(
        OPENROUTER_URL,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": model,
            "input_audio": {
                "data": audio_b64,
                "format": "mp3",
            },
        },
        timeout=120,
    )

    if resp.status_code != 200:
        raise RuntimeError(f"Transcription failed ({resp.status_code}): {resp.text}")

    data = resp.json()
    cost = data.get("usage", {}).get("cost", 0)
    if cost:
        console.log(f"Transcription cost: [green]${cost:.4f}[/green]")

    return {
        "segments": [],
        "full_text": data["text"],
        "language": "en",
        "source": f"openrouter/{model}",
    }


def fetch_transcript(video_id: str, languages: list[str] | None = None, api_key: str | None = None, no_cache: bool = False, meta: dict | None = None) -> dict:
    if not no_cache:
        cached = cache_get(video_id, "transcript")
        if cached:
            console.log(f"[green]✓[/green] Transcript loaded from cache")
            return cached

    try:
        result = fetch_transcript_captions(video_id, languages)
        console.log(f"[green]✓[/green] Got captions from YouTube [dim](free)[/dim]")
    except Exception as caption_err:
        if not api_key:
            raise click.ClickException(
                f"No captions available ({caption_err}). "
                "Set OPENROUTER_API_KEY to enable audio transcription fallback."
            )

        console.log(f"[yellow]⚠[/yellow] No captions — falling back to audio transcription")

        with tempfile.TemporaryDirectory() as tmpdir:
            audio_path = download_audio(video_id, tmpdir)
            result = transcribe_audio(audio_path, api_key)

    cache_set(video_id, "transcript", result, meta=meta)
    return result


def _call_llm(messages: list, model: str, json_mode: bool = False, lf: Langfuse | None = None, span_name: str = "analyze") -> tuple[str, dict]:
    generation = None
    if lf:
        generation = lf.start_observation(
            as_type="generation",
            name=span_name,
            model=model,
            input=messages,
            metadata={"base_url": ANALYSIS_BASE_URL},
        )

    body = {"model": model, "messages": messages}
    if json_mode:
        body["response_format"] = {"type": "json_object"}

    resp = requests.post(
        ANALYSIS_BASE_URL,
        headers={
            "Authorization": f"Bearer {ANALYSIS_API_KEY}",
            "Content-Type": "application/json",
        },
        json=body,
        timeout=60,
    )

    if resp.status_code != 200:
        if generation:
            generation.update(output={"error": resp.text}, level="ERROR")
            generation.end()
        raise RuntimeError(f"Analysis failed ({resp.status_code}): {resp.text}")

    resp_data = resp.json()
    content = resp_data["choices"][0]["message"]["content"]
    usage = resp_data.get("usage", {})

    if generation:
        generation.update(
            output=content,
            usage_details={
                "input": usage.get("prompt_tokens", 0),
                "output": usage.get("completion_tokens", 0),
                "total": usage.get("total_tokens", 0),
            },
        )
        generation.end()

    return content, usage


def analyze_transcript(transcript_text: str, metadata: dict | None = None, model: str = DEFAULT_ANALYSIS_MODEL, lf: Langfuse | None = None, video_id: str | None = None, no_cache: bool = False, custom_prompt: str | None = None, verify: bool = False) -> dict:
    if video_id and not no_cache:
        model_tag = model.replace("/", "_")
        cached = cache_get(video_id, f"analysis.{model_tag}")
        if cached:
            console.log(f"[green]✓[/green] Analysis loaded from cache [dim]({model})[/dim]")
            return cached

    context = ""
    if metadata:
        context = f"Title: {metadata['title']}\nAuthor: {metadata['author']}\n\n"

    system_prompt = custom_prompt or ANALYSIS_PROMPT
    use_json_mode = custom_prompt is None

    transcript_msg = f"{context}TRANSCRIPT:\n{transcript_text}"

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": transcript_msg},
    ]

    # Pass 1
    content, usage = _call_llm(messages, model, json_mode=use_json_mode, lf=lf, span_name="analyze_pass1")
    console.log(f"[green]✓[/green] Pass 1 complete [dim]({usage.get('total_tokens', '?')} tokens)[/dim]")

    # Pass 2 — verification
    # Keep system + first user message identical to pass 1 for prompt cache hit,
    # then add assistant response + new user turn for verification.
    if verify:
        console.log("Verifying analysis...")
        verify_suffix = (
            "Please verify if this analysis is correct and complete with respect to the transcript. "
            "Start with your reasoning about what's accurate, what's missing, and what's wrong. "
            "Then output the final corrected result."
        )
        if use_json_mode:
            verify_suffix += "\n\nOutput the corrected result as JSON in the same schema."

        verify_messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": transcript_msg},
            {"role": "assistant", "content": content},
            {"role": "user", "content": verify_suffix},
        ]

        content, v_usage = _call_llm(verify_messages, model, json_mode=use_json_mode, lf=lf, span_name="analyze_verify")
        cached = v_usage.get("prompt_tokens_details", {}).get("cached_tokens", 0)
        console.log(f"[green]✓[/green] Verification complete [dim]({v_usage.get('total_tokens', '?')} tokens, {cached} cached)[/dim]")

    if use_json_mode:
        parsed = json.loads(content)
        if video_id:
            model_tag = model.replace("/", "_")
            cache_set(video_id, f"analysis.{model_tag}", parsed, meta=metadata)
        return parsed

    return {"raw": content}


def format_timestamp(seconds: float) -> str:
    mins, secs = divmod(int(seconds), 60)
    hours, mins = divmod(mins, 60)
    return f"{hours:02d}:{mins:02d}:{secs:02d}" if hours else f"{mins:02d}:{secs:02d}"


def render_analysis(analysis: dict, meta: dict | None):
    title = meta["title"] if meta else "Video Analysis"
    author = meta["author"] if meta else None

    subtitle = f"by {author}" if author else None
    out.print(Panel(analysis["summary"], title=title, subtitle=subtitle, border_style="blue", padding=(1, 2)))

    out.print()

    # Key points
    out.print("[bold]Key Points[/bold]")
    for i, point in enumerate(analysis.get("key_points", []), 1):
        out.print(f"  [dim]{i}.[/dim] {point}")

    # Quotes
    quotes = analysis.get("notable_quotes", [])
    if quotes:
        out.print()
        out.print("[bold]Notable Quotes[/bold]")
        for q in quotes:
            out.print(f'  [italic]"{q}"[/italic]')

    # Action items
    actions = analysis.get("action_items", [])
    if actions:
        out.print()
        out.print("[bold]Action Items[/bold]")
        for item in actions:
            out.print(f"  [green]→[/green] {item}")

    # Footer table
    out.print()
    tbl = Table(box=box.SIMPLE, show_header=False, padding=(0, 2))
    tbl.add_column(style="dim")
    tbl.add_column()
    tbl.add_row("Topics", ", ".join(analysis.get("topics", [])))
    tbl.add_row("Sentiment", analysis.get("sentiment", "unknown"))
    out.print(tbl)


def render_transcript(result: dict, timestamps: bool):
    source_label = result["source"].replace("_", " ").title()
    console.log(f"Source: [cyan]{source_label}[/cyan] · [dim]{len(result['full_text']):,} chars[/dim]")
    out.print()

    if timestamps and result["segments"]:
        for seg in result["segments"]:
            ts = format_timestamp(seg["start"])
            out.print(f"[dim]{ts}[/dim]  {seg['text']}")
    else:
        out.print(result["full_text"])


@click.command()
@click.argument("video", required=False)
@click.option("--open-cache", "-o", is_flag=True, help="Open cache folder in file manager")
@click.option("--lang", "-l", multiple=True, help="Language codes to try (default: en)")
@click.option("--json-output", "-j", is_flag=True, help="Output as JSON")
@click.option("--metadata/--no-metadata", "-m/-M", default=True, help="Include metadata")
@click.option("--timestamps/--no-timestamps", "-t/-T", default=False, help="Include timestamps in text output")
@click.option("--proxy", "-p", is_flag=True, help="Force proxy usage (requires EVOMI_USER/EVOMI_PASS)")
@click.option("--force-transcribe", "-f", is_flag=True, help="Skip captions, always use audio transcription")
@click.option("--model", default=DEFAULT_TRANSCRIPTION_MODEL, help="Transcription model for fallback")
@click.option("--analyze", "-a", is_flag=True, help="Analyze transcript with Gemini via Bifrost")
@click.option("--verify", "-v", is_flag=True, help="Double-pass: analyze then verify (uses prompt cache)")
@click.option("--prompt", "-q", default=None, help="Custom analysis prompt (replaces default)")
@click.option("--analysis-model", default=DEFAULT_ANALYSIS_MODEL, help="Model for analysis")
@click.option("--no-cache", is_flag=True, help="Skip cache, fetch fresh data")
def main(video, open_cache, lang, json_output, metadata, timestamps, proxy, force_transcribe, model, analyze, verify, prompt, analysis_model, no_cache):
    """Analyze a video by extracting its transcript.

    VIDEO can be a YouTube URL or video ID.

    Uses YouTube captions when available (free), falls back to audio
    transcription via OpenRouter when captions are missing.
    """
    if open_cache:
        os.makedirs(CACHE_DIR, exist_ok=True)
        subprocess.run(["xdg-open", CACHE_DIR])
        return

    if not video:
        raise click.UsageError("Missing argument 'VIDEO'. Use --open-cache or provide a video URL/ID.")

    video_id = extract_video_id(video)
    api_key = os.environ.get("OPENROUTER_API_KEY")

    if proxy and not os.environ.get("EVOMI_USER"):
        raise click.ClickException("--proxy requires EVOMI_USER and EVOMI_PASS env vars")

    if force_transcribe and not api_key:
        raise click.ClickException("--force-transcribe requires OPENROUTER_API_KEY env var")

    lf = Langfuse()

    console.log(f"[bold]Video:[/bold] [cyan]{video_id}[/cyan]")

    languages = list(lang) if lang else None

    meta = None
    if metadata:
        if not no_cache:
            meta = cache_get(video_id, "metadata")
            if meta:
                console.log(f"[bold]Title:[/bold] {meta['title']} [dim](cached)[/dim]")
                console.log(f"[bold]Author:[/bold] {meta['author']}")
        if not meta:
            meta = fetch_metadata(video_id)
            if meta:
                cache_set(video_id, "metadata", meta, meta=meta)
                console.log(f"[bold]Title:[/bold] {meta['title']}")
                console.log(f"[bold]Author:[/bold] {meta['author']}")

    if force_transcribe:
        console.log("[yellow]Forced transcription mode[/yellow] — downloading audio...")
        with tempfile.TemporaryDirectory() as tmpdir:
            audio_path = download_audio(video_id, tmpdir)
            result = transcribe_audio(audio_path, api_key, model)
            cache_set(video_id, "transcript", result, meta=meta)
    else:
        result = fetch_transcript(video_id, languages, api_key, no_cache=no_cache, meta=meta)

    if prompt or verify:
        analyze = True

    analysis = None
    if analyze:
        console.log(f"Analyzing with [cyan]{analysis_model}[/cyan]{'  [dim](+ verify)[/dim]' if verify else ''}...")
        analysis = analyze_transcript(result["full_text"], meta, analysis_model, lf=lf, video_id=video_id, no_cache=no_cache or bool(prompt), custom_prompt=prompt, verify=verify)

    # Output
    if json_output:
        output = {"video_id": video_id}
        if meta:
            output["metadata"] = meta
        output["transcript"] = result
        if analysis:
            output["analysis"] = analysis
        out.print_json(json.dumps(output, ensure_ascii=False))
    elif analysis and "raw" in analysis:
        out.print(analysis["raw"])
    elif analysis:
        render_analysis(analysis, meta)
    else:
        render_transcript(result, timestamps)


if __name__ == "__main__":
    main()

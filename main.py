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
from youtube_transcript_api.proxies import GenericProxyConfig

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


ANALYSIS_PROMPT = """you are grug-brained video watcher. grug watch transcript, grug say what actually matters. no corporate fluff. no "the hosts discuss". grug say what they SAID and whether grug thinks it's smart or dumb.

grug rules:
- be opinionated. if take is mid, say it's mid. if it's actually interesting, say why.
- use plain language. no "explores the intersection of" garbage.
- extract SPECIFICS: every dollar amount, tool name, metric, company, book, person named. if they said "$70/month on MidJourney" or "SpaceX at $1.8T" — that goes in. vague summaries are useless.
- key points should be things someone would actually repeat to a friend, not boardroom summaries. include the numbers and names.
- quotes should be the spicy ones, not the obvious ones.
- if someone said something wrong, naive, or factually off — call it out directly.
- focus on the USEFUL stuff — what can a dev, marketer, or entrepreneur actually DO with this info to make money or get ahead?
- BE CYNICAL. question the speakers' incentives. are they selling something? pumping a bag? building an audience? do they benefit if you follow their advice?
- if the transcript has timestamps, reference them. if not, estimate approximate timestamps for key moments.

Output JSON with these fields:
- summary: 2-3 sentence TL;DW — what's the actual take, written like you're texting a friend
- key_points: array of objects, each with "point" (the takeaway with specific names/numbers), "timestamp" (approximate MM:SS or "~MM:SS" if estimated), and optionally "bs_flag" (string, only if the claim is dubious — say why)
- tools_mentioned: array of objects with "name", "verdict" (what the speakers actually think of it — "daily driver", "canceled", "dead to me", etc.), and "detail" (1 sentence of context)
- topics: array of topic tags
- sentiment: overall tone (positive/negative/neutral/mixed)
- notable_quotes: array of objects with "quote" (exact words) and "timestamp" (MM:SS or approximate)
- bullshit_meter: object with "level" (one of: "legit", "mostly_legit", "mixed_bag", "heavy_spin", "pure_hype") and "reasoning" (1-2 sentences on speaker incentives, what they're selling, and how much to actually trust)
- who_cares: object with keys like "devs", "marketers", "founders", etc — each value is a 1-2 sentence blurb on why this video matters to THEM specifically, or null if it doesn't
- action_items: array of concrete, specific things you can go do RIGHT NOW to make money or get an edge (empty array if it's all talk — "consider exploring X" is NOT an action item, "go sign up for Y and use it to do Z" IS)"""


def is_youtube_url(url_or_id: str) -> bool:
    """Return True if the input looks like a YouTube URL or bare video ID."""
    patterns = [
        r"(?:youtube\.com/watch\?v=|youtu\.be/|youtube\.com/embed/|youtube\.com/shorts/)[a-zA-Z0-9_-]{11}",
        r"^[a-zA-Z0-9_-]{11}$",
    ]
    return any(re.search(p, url_or_id) for p in patterns)


def extract_video_id(url_or_id: str) -> str | None:
    """Extract YouTube video ID. Returns None for non-YouTube URLs."""
    patterns = [
        r"(?:youtube\.com/watch\?v=|youtu\.be/|youtube\.com/embed/|youtube\.com/shorts/)([a-zA-Z0-9_-]{11})",
        r"^([a-zA-Z0-9_-]{11})$",
    ]
    for pattern in patterns:
        match = re.search(pattern, url_or_id)
        if match:
            return match.group(1)
    return None


def url_cache_key(url: str) -> str:
    """Generate a short cache key from a URL hash."""
    import hashlib
    return hashlib.sha256(url.encode()).hexdigest()[:16]


def get_proxy_config():
    proxy_url = get_proxy_url()
    if not proxy_url:
        return None
    return GenericProxyConfig(http_url=proxy_url, https_url=proxy_url)


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

    ytt = YouTubeTranscriptApi(proxy_config=proxy_config)
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


def get_proxy_url() -> str | None:
    user = os.environ.get("EVOMI_USER")
    password = os.environ.get("EVOMI_PASS")
    if not user or not password:
        return None
    return f"http://{user}:{password}@core-residential.evomi.com:1000"


def download_audio(url: str, tmpdir: str, use_proxy: bool = False) -> str:
    """Download audio from any URL via yt-dlp."""
    out_path = os.path.join(tmpdir, "audio.%(ext)s")
    cmd = [
        "yt-dlp", "-x",
        "--audio-format", "mp3",
        "--audio-quality", "9",
        "--max-filesize", "100M",
        "-o", out_path,
    ]
    if use_proxy:
        proxy_url = get_proxy_url()
        if proxy_url:
            cmd.extend(["--proxy", proxy_url])
            console.log("[dim]Using Evomi proxy[/dim]")
    cmd.append(url)
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


def fetch_transcript_yt(video_id: str, url: str, languages: list[str] | None = None, api_key: str | None = None, no_cache: bool = False, meta: dict | None = None, use_proxy: bool = False) -> dict:
    """Fetch transcript for a YouTube video: captions first, then yt-dlp audio fallback."""
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
            audio_path = download_audio(url, tmpdir, use_proxy=use_proxy)
            result = transcribe_audio(audio_path, api_key)

    cache_set(video_id, "transcript", result, meta=meta)
    return result


def fetch_transcript_generic(url: str, cache_key: str, api_key: str, no_cache: bool = False, meta: dict | None = None, use_proxy: bool = False) -> dict:
    """Fetch transcript for a non-YouTube URL via yt-dlp audio + transcription."""
    if not no_cache:
        cached = cache_get(cache_key, "transcript")
        if cached:
            console.log(f"[green]✓[/green] Transcript loaded from cache")
            return cached

    console.log(f"Downloading audio via yt-dlp...")
    with tempfile.TemporaryDirectory() as tmpdir:
        audio_path = download_audio(url, tmpdir, use_proxy=use_proxy)
        result = transcribe_audio(audio_path, api_key)

    cache_set(cache_key, "transcript", result, meta=meta)
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
        if isinstance(parsed, list) and len(parsed) == 1:
            parsed = parsed[0]
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
        if isinstance(point, dict):
            ts = f"[dim][{point.get('timestamp', '?')}][/dim] " if point.get("timestamp") else ""
            text = point.get("point", "")
            bs_flag = point.get("bs_flag")
            out.print(f"  [dim]{i}.[/dim] {ts}{text}")
            if bs_flag:
                out.print(f"     [red]⚠ {bs_flag}[/red]")
        else:
            out.print(f"  [dim]{i}.[/dim] {point}")

    # Tools mentioned
    tools = analysis.get("tools_mentioned", [])
    if tools:
        out.print()
        out.print("[bold]Tools Mentioned[/bold]")
        for t in tools:
            if isinstance(t, dict):
                verdict_colors = {"daily driver": "green", "canceled": "red", "dead to me": "red", "occasional": "yellow"}
                verdict = t.get("verdict", "")
                color = next((c for k, c in verdict_colors.items() if k in verdict.lower()), "cyan")
                out.print(f"  [bold]{t.get('name', '?')}[/bold] — [{color}]{verdict}[/{color}]")
                if t.get("detail"):
                    out.print(f"    [dim]{t['detail']}[/dim]")
            else:
                out.print(f"  {t}")

    # Quotes
    quotes = analysis.get("notable_quotes", [])
    if quotes:
        out.print()
        out.print("[bold]Notable Quotes[/bold]")
        for q in quotes:
            if isinstance(q, dict):
                ts = f" [dim][{q.get('timestamp', '?')}][/dim]" if q.get("timestamp") else ""
                out.print(f'  [italic]"{q.get("quote", q)}"[/italic]{ts}')
            else:
                out.print(f'  [italic]"{q}"[/italic]')

    # Bullshit meter
    bs = analysis.get("bullshit_meter", {})
    if bs:
        level = bs.get("level", "?")
        reasoning = bs.get("reasoning", "")
        level_colors = {"legit": "green", "mostly_legit": "green", "mixed_bag": "yellow", "heavy_spin": "red", "pure_hype": "red bold"}
        color = level_colors.get(level, "white")
        label = level.replace("_", " ")
        out.print()
        out.print(f"[bold]Bullshit Meter[/bold]: [{color}]{label}[/{color}]")
        if reasoning:
            out.print(f"  [dim]{reasoning}[/dim]")

    # Who cares
    who_cares = analysis.get("who_cares", {})
    if who_cares:
        out.print()
        out.print("[bold]Who Cares[/bold]")
        for role, blurb in who_cares.items():
            if blurb:
                out.print(f"  [yellow]{role}[/yellow]: {blurb}")

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
@click.option("--chat", "-c", is_flag=True, help="Chat about the video after analysis")
def main(video, open_cache, lang, json_output, metadata, timestamps, proxy, force_transcribe, model, analyze, verify, prompt, analysis_model, no_cache, chat):
    """Analyze a video by extracting its transcript.

    VIDEO can be a YouTube URL, video ID, or any video URL supported by yt-dlp
    (Twitter/X, TikTok, Instagram, direct mp4 links, etc.).

    For YouTube: tries captions first (free), then falls back to audio transcription.
    For other URLs: downloads audio via yt-dlp and transcribes via OpenRouter.
    """
    if open_cache:
        os.makedirs(CACHE_DIR, exist_ok=True)
        subprocess.run(["xdg-open", CACHE_DIR])
        return

    if not video:
        raise click.UsageError("Missing argument 'VIDEO'. Use --open-cache or provide a video URL/ID.")

    api_key = os.environ.get("OPENROUTER_API_KEY")
    youtube = is_youtube_url(video)
    video_id = extract_video_id(video) if youtube else None
    # For YouTube bare IDs, build the full URL for yt-dlp
    yt_url = f"https://www.youtube.com/watch?v={video_id}" if (youtube and video_id and not video.startswith("http")) else video
    # Cache key: YouTube video ID or hash of URL
    c_key = video_id or url_cache_key(video)

    if proxy and not get_proxy_url():
        raise click.ClickException("--proxy requires EVOMI_USER and EVOMI_PASS env vars")

    if force_transcribe and not api_key:
        raise click.ClickException("--force-transcribe requires OPENROUTER_API_KEY env var")

    if not youtube and not api_key:
        raise click.ClickException("Non-YouTube URLs require OPENROUTER_API_KEY for audio transcription.")

    lf = Langfuse()

    console.log(f"[bold]Video:[/bold] [cyan]{video_id or video}[/cyan]")

    languages = list(lang) if lang else None

    # Metadata: only fetch for YouTube
    meta = None
    if metadata and youtube and video_id:
        if not no_cache:
            meta = cache_get(c_key, "metadata")
            if meta:
                console.log(f"[bold]Title:[/bold] {meta['title']} [dim](cached)[/dim]")
                console.log(f"[bold]Author:[/bold] {meta['author']}")
        if not meta:
            meta = fetch_metadata(video_id)
            if meta:
                cache_set(c_key, "metadata", meta, meta=meta)
                console.log(f"[bold]Title:[/bold] {meta['title']}")
                console.log(f"[bold]Author:[/bold] {meta['author']}")

    if force_transcribe:
        console.log("[yellow]Forced transcription mode[/yellow] — downloading audio...")
        with tempfile.TemporaryDirectory() as tmpdir:
            audio_path = download_audio(yt_url, tmpdir, use_proxy=proxy)
            result = transcribe_audio(audio_path, api_key, model)
            cache_set(c_key, "transcript", result, meta=meta)
    elif youtube and video_id:
        result = fetch_transcript_yt(video_id, yt_url, languages, api_key, no_cache=no_cache, meta=meta, use_proxy=proxy)
    else:
        result = fetch_transcript_generic(video, c_key, api_key, no_cache=no_cache, meta=meta, use_proxy=proxy)

    if prompt or verify or chat:
        analyze = True

    analysis = None
    if analyze:
        console.log(f"Analyzing with [cyan]{analysis_model}[/cyan]{'  [dim](+ verify)[/dim]' if verify else ''}...")
        analysis = analyze_transcript(result["full_text"], meta, analysis_model, lf=lf, video_id=c_key, no_cache=no_cache or bool(prompt), custom_prompt=prompt, verify=verify)

    # Output
    if json_output:
        output = {"video_id": video_id} if video_id else {"url": video}
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

    if chat:
        start_chat(result["full_text"], analysis, meta, analysis_model)


def start_chat(transcript_text: str, analysis: dict | None, meta: dict | None, model_name: str):
    from llm.default_plugins.openai_models import Chat

    llm_model = Chat(
        model_id="va-chat",
        model_name=model_name,
        api_base=ANALYSIS_BASE_URL.replace("/chat/completions", ""),
    )
    llm_model.key = ANALYSIS_API_KEY

    title = meta["title"] if meta else "this video"
    system = f"You have the full transcript of \"{title}\" and its analysis. Answer questions about it. Be direct and specific — quote the transcript when relevant. Stay grug-brained: no fluff, opinions welcome."

    context = f"TRANSCRIPT:\n{transcript_text}"
    if analysis:
        context += f"\n\nANALYSIS:\n{json.dumps(analysis, ensure_ascii=False)}"

    conv = llm_model.conversation()
    # seed with transcript context
    r = conv.prompt(context, system=system)
    r.text()  # consume but don't print the seed response

    out.print()
    out.print("[bold]Chat mode[/bold] — ask anything about the video. [dim]Ctrl+C to exit.[/dim]")
    out.print()

    try:
        while True:
            try:
                question = input("you> ").strip()
            except EOFError:
                break
            if not question:
                continue
            response = conv.prompt(question)
            for chunk in response:
                print(chunk, end="", flush=True)
            print()
            print()
    except KeyboardInterrupt:
        out.print("\n[dim]bye[/dim]")


if __name__ == "__main__":
    main()

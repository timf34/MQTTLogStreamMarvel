#!/usr/bin/env python3
"""Podscript - Transcribe podcasts and YouTube videos using ElevenLabs Scribe API."""

import argparse
import html
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from urllib.parse import urlparse, parse_qs

import feedparser
import requests
from dotenv import load_dotenv

# Constants
ELEVENLABS_API_URL = "https://api.elevenlabs.io/v1/speech-to-text"
PAUSE_THRESHOLD = 1.0  # seconds - gap that triggers new segment


# Dataclasses
@dataclass
class Episode:
    title: str
    audio_url: str
    publish_date: str
    description: str
    duration: str


@dataclass
class TranscriptSegment:
    speaker: str
    text: str
    start: float
    end: float


# ── Helpers ──────────────────────────────────────────────────────────────────


def format_timestamp(seconds: float) -> str:
    """Format seconds as M:SS or H:MM:SS."""
    total = int(seconds)
    h = total // 3600
    m = (total % 3600) // 60
    s = total % 60
    if h > 0:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m}:{s:02d}"


def sanitize_filename(name: str) -> str:
    """Strip non-alphanumeric chars, collapse whitespace to hyphens, max 80 chars."""
    name = re.sub(r"[^a-zA-Z0-9\s-]", "", name)
    name = re.sub(r"\s+", "-", name).strip("-")
    return name[:80]


def clean_html(text: str) -> str:
    """Strip HTML tags and unescape entities."""
    text = re.sub(r"<[^>]*>", "", text)
    return html.unescape(text).strip()


def format_duration(duration) -> str:
    """Normalise an itunes:duration value (seconds int, HH:MM:SS string, etc.)."""
    if not duration:
        return ""
    if isinstance(duration, (int, float)):
        return format_timestamp(duration)
    duration = str(duration)
    if ":" in duration:
        return duration
    try:
        return format_timestamp(int(duration))
    except ValueError:
        return duration


# ── YouTube ──────────────────────────────────────────────────────────────────


def is_youtube_url(url: str) -> bool:
    return bool(re.search(r"(?:youtube\.com/watch|youtu\.be/|youtube\.com/shorts/)", url))


def get_yt_dlp_cmd() -> str:
    """Return the yt-dlp invocation that works on this system."""
    for cmd in ["yt-dlp", "python -m yt_dlp"]:
        try:
            subprocess.run(
                [*cmd.split(), "--version"],
                capture_output=True,
                check=True,
            )
            return cmd
        except (subprocess.CalledProcessError, FileNotFoundError):
            continue
    print(
        "Error: yt-dlp is not installed. Install it with: pip install yt-dlp\n"
        "You also need ffmpeg installed for audio extraction.",
        file=sys.stderr,
    )
    sys.exit(1)


def clean_youtube_url(url: str) -> str:
    """Strip timestamp parameters that can cause shell issues."""
    try:
        from urllib.parse import urlparse, urlencode, parse_qs

        parsed = urlparse(url)
        params = parse_qs(parsed.query)
        params.pop("t", None)
        clean_query = urlencode(params, doseq=True)
        return parsed._replace(query=clean_query).geturl()
    except Exception:
        return url


def download_youtube_audio(url: str) -> dict:
    """Download audio from YouTube, return dict with title, channel, duration, audio_path."""
    yt_dlp = get_yt_dlp_cmd()
    clean_url = clean_youtube_url(url)

    # Get metadata
    print("Fetching video info...\n")
    result = subprocess.run(
        [*yt_dlp.split(), "--no-download", "--dump-json", clean_url],
        capture_output=True,
        text=True,
        timeout=60,
    )
    if result.returncode != 0:
        print(f"Error fetching video info: {result.stderr}", file=sys.stderr)
        sys.exit(1)

    info = json.loads(result.stdout)
    title = info.get("title", "Untitled")
    channel = info.get("channel") or info.get("uploader") or "Unknown"
    duration = info.get("duration", 0)

    print(f"Title: {title}")
    print(f"Channel: {channel}")
    print(f"Duration: {format_duration(duration)}\n")

    # Download audio
    temp_base = os.path.join(tempfile.gettempdir(), f"podscript-{os.getpid()}")
    temp_output = f"{temp_base}.%(ext)s"

    print("Downloading audio...")
    subprocess.run(
        [*yt_dlp.split(), "-x", "--audio-format", "mp3", "--audio-quality", "0", "-o", temp_output, clean_url],
        timeout=600,
        check=True,
    )

    audio_path = f"{temp_base}.mp3"
    if not os.path.exists(audio_path):
        print(f"Error: Audio download failed - expected file at {audio_path}", file=sys.stderr)
        sys.exit(1)

    size_mb = os.path.getsize(audio_path) / (1024 * 1024)
    print(f"Downloaded: {size_mb:.1f} MB\n")

    return {"title": title, "channel": channel, "duration": duration, "audio_path": audio_path}


# ── Apple Podcasts ────────────────────────────────────────────────────────


def parse_apple_podcasts_url(url: str) -> tuple[str, str | None] | None:
    """
    If url is an Apple Podcasts link, return (podcast_id, episode_id or None).
    Otherwise return None.
    """
    parsed = urlparse(url)
    if not parsed.hostname or "podcasts.apple.com" not in parsed.hostname:
        return None
    # Extract podcast ID from path like /gb/podcast/some-name/id842818711
    m = re.search(r"/id(\d+)", parsed.path)
    if not m:
        return None
    podcast_id = m.group(1)
    # Extract episode ID from query param ?i=1000588160381
    params = parse_qs(parsed.query)
    episode_id = params.get("i", [None])[0]
    return podcast_id, episode_id


def resolve_apple_podcasts_url(url: str) -> tuple[str, str | None]:
    """
    Resolve an Apple Podcasts URL to its RSS feed URL.
    Returns (feed_url, episode_id or None).
    """
    result = parse_apple_podcasts_url(url)
    if result is None:
        raise ValueError("Not an Apple Podcasts URL")
    podcast_id, episode_id = result

    print(f"Resolving Apple Podcasts ID {podcast_id} to RSS feed...")
    resp = requests.get(
        f"https://itunes.apple.com/lookup?id={podcast_id}&entity=podcast",
        timeout=15,
    )
    resp.raise_for_status()
    data = resp.json()
    results = data.get("results", [])
    if not results:
        raise RuntimeError(f"No podcast found for Apple Podcasts ID {podcast_id}")
    feed_url = results[0].get("feedUrl")
    if not feed_url:
        raise RuntimeError(f"No RSS feed URL found for Apple Podcasts ID {podcast_id}")
    print(f"Found RSS feed: {feed_url}\n")
    return feed_url, episode_id


def scrape_apple_episode_info(apple_url: str) -> dict | None:
    """
    Scrape episode title and audio URL from an Apple Podcasts page.
    Returns dict with 'title' and optionally 'audio_url', or None on failure.
    """
    try:
        resp = requests.get(apple_url, timeout=15, headers={"User-Agent": "Mozilla/5.0"})
        resp.raise_for_status()
        page = resp.text
    except Exception:
        return None

    info = {}
    # Extract title from og:title meta tag (e.g. "The Economics of Carbon Removal ...")
    m = re.search(r'<meta\s[^>]*property="og:title"\s[^>]*content="([^"]*)"', page)
    if not m:
        m = re.search(r'<meta\s[^>]*content="([^"]*)"\s[^>]*property="og:title"', page)
    if m:
        info["title"] = html.unescape(m.group(1))

    # Extract audio URL from streamUrl in embedded JSON
    m = re.search(r'"streamUrl"\s*:\s*"(https?://[^"]+\.mp3[^"]*)"', page)
    if m:
        info["audio_url"] = m.group(1)

    return info if info else None


# ── RSS ──────────────────────────────────────────────────────────────────────


def parse_feed(feed_url: str) -> tuple[str, list[Episode]]:
    """Parse an RSS feed, return (podcast_name, episodes)."""
    feed = feedparser.parse(feed_url)

    if feed.bozo and not feed.entries:
        raise RuntimeError(f"Failed to parse feed: {feed.bozo_exception}")

    podcast_name = getattr(feed.feed, "title", "Unknown Podcast")
    episodes: list[Episode] = []

    for entry in feed.entries:
        # Find audio URL from enclosures
        audio_url = ""
        for enc in getattr(entry, "enclosures", []):
            href = enc.get("href") or enc.get("url", "")
            if href:
                audio_url = href
                break
        if not audio_url:
            continue

        title = getattr(entry, "title", "Untitled Episode")
        pub_date = getattr(entry, "published", "")
        summary = getattr(entry, "summary", "") or getattr(entry, "description", "")
        description = clean_html(summary)[:200]
        if len(clean_html(summary)) > 200:
            description += "..."
        raw_duration = getattr(entry, "itunes_duration", "")
        duration = format_duration(raw_duration)

        episodes.append(Episode(
            title=title,
            audio_url=audio_url,
            publish_date=pub_date,
            description=description,
            duration=duration,
        ))

    return podcast_name, episodes


# ── ElevenLabs transcription ────────────────────────────────────────────────


def transcribe(source: str, *, is_file: bool = False) -> tuple[list[TranscriptSegment], float]:
    """
    Transcribe audio via ElevenLabs Scribe API.

    Args:
        source: Either a cloud URL or a local file path.
        is_file: True if source is a local file path.

    Returns:
        (segments, duration_seconds)
    """
    api_key = os.environ.get("ELEVENLABS_API_KEY", "")
    if not api_key:
        print("Error: ELEVENLABS_API_KEY not found in environment.", file=sys.stderr)
        sys.exit(1)

    data = {
        "model_id": "scribe_v1",
        "diarize": "true",
        "timestamps_granularity": "word",
    }
    files = None

    if is_file:
        files = {"file": ("audio.mp3", open(source, "rb"), "audio/mpeg")}
    else:
        data["cloud_storage_url"] = source

    resp = requests.post(
        ELEVENLABS_API_URL,
        headers={"xi-api-key": api_key},
        data=data,
        files=files,
        timeout=600,
    )

    # Close the file handle if we opened one
    if files and "file" in files:
        files["file"][1].close()

    if resp.status_code != 200:
        print(f"ElevenLabs API error ({resp.status_code}): {resp.text}", file=sys.stderr)
        sys.exit(1)

    body = resp.json()
    words = body.get("words", [])
    segments = group_into_segments(words)
    duration = words[-1]["end"] if words else 0.0

    return segments, duration


def group_into_segments(words: list[dict]) -> list[TranscriptSegment]:
    """Group words into segments by speaker changes and pauses > PAUSE_THRESHOLD."""
    if not words:
        return []

    segments: list[TranscriptSegment] = []
    cur: TranscriptSegment | None = None

    for w in words:
        speaker = w.get("speaker_id") or "speaker_0"
        start_new = (
            cur is None
            or cur.speaker != speaker
            or (w["start"] - cur.end > PAUSE_THRESHOLD)
        )

        if start_new:
            if cur is not None:
                segments.append(cur)
            cur = TranscriptSegment(speaker=speaker, text=w["text"], start=w["start"], end=w["end"])
        else:
            cur.text += " " + w["text"]
            cur.end = w["end"]

    if cur is not None:
        segments.append(cur)

    # Clean up whitespace
    for seg in segments:
        seg.text = re.sub(r"\s+", " ", seg.text).strip()

    return segments


# ── Markdown generation ─────────────────────────────────────────────────────


def speaker_name(speaker_id: str) -> str:
    """Convert speaker_0 → Speaker 1, etc."""
    m = re.match(r"speaker_(\d+)", speaker_id)
    if m:
        return f"Speaker {int(m.group(1)) + 1}"
    return speaker_id


def generate_markdown(
    title: str,
    podcast_name: str,
    segments: list[TranscriptSegment],
    duration_secs: float,
) -> str:
    lines = [
        f"# {title}",
        "",
        f"**Podcast:** {podcast_name}",
        f"**Date:** {datetime.now().month}/{datetime.now().day}/{datetime.now().year}",
        f"**Duration:** {format_timestamp(duration_secs)}",
        "",
        "---",
        "",
    ]

    current_speaker = ""
    for seg in segments:
        name = speaker_name(seg.speaker)
        if name != current_speaker:
            lines.append(f"## {name}")
            current_speaker = name
        lines.append(f"[{format_timestamp(seg.start)}] {seg.text}")
        lines.append("")

    return "\n".join(lines)


# ── CLI ──────────────────────────────────────────────────────────────────────


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="podscript",
        description="Transcribe podcasts and YouTube videos using ElevenLabs Scribe API.",
    )
    parser.add_argument("url", help="RSS feed URL or YouTube video URL")
    parser.add_argument("--episode", type=int, metavar="N", help="Transcribe episode N (1 = most recent)")
    parser.add_argument("--search", metavar="QUERY", help="Search episodes by title/description")
    parser.add_argument("--latest", action="store_true", help="Transcribe the most recent episode (default)")
    parser.add_argument("--list", action="store_true", dest="list_episodes", help="List episodes without transcribing")
    parser.add_argument("--output", metavar="FILE", help="Output filename (default: auto-generated)")
    return parser


def require_api_key():
    """Exit if ELEVENLABS_API_KEY is not set."""
    if not os.environ.get("ELEVENLABS_API_KEY", ""):
        print("Error: ELEVENLABS_API_KEY not found in environment.", file=sys.stderr)
        print("Make sure you have a .env file with your API key.", file=sys.stderr)
        sys.exit(1)


def main():
    load_dotenv()

    parser = build_parser()
    args = parser.parse_args()

    url: str = args.url

    # ── YouTube path ─────────────────────────────────────────────────────
    if is_youtube_url(url):
        require_api_key()
        print("\nDetected YouTube URL\n")
        yt = download_youtube_audio(url)

        print("Starting transcription...")
        print("This may take several minutes depending on video length.\n")

        try:
            t0 = time.time()
            segments, duration = transcribe(yt["audio_path"], is_file=True)
            elapsed = int(time.time() - t0)
        finally:
            # Clean up temp file
            try:
                os.unlink(yt["audio_path"])
            except OSError:
                pass

        print(f"\nTranscription complete in {elapsed} seconds.")
        print(f"Duration: {format_timestamp(duration)}")
        print(f"Segments: {len(segments)}")

        md = generate_markdown(yt["title"], yt["channel"], segments, duration)
        filename = args.output or f"{sanitize_filename(yt['title'])}.md"
        Path(filename).write_text(md, encoding="utf-8")
        print(f"\nSaved to: {filename}")
        return

    # ── Resolve Apple Podcasts URLs to RSS ──────────────────────────────
    apple_episode_id = None
    original_url = url
    if parse_apple_podcasts_url(url):
        url, apple_episode_id = resolve_apple_podcasts_url(url)

    # ── Podcast RSS path ─────────────────────────────────────────────────
    print(f"Fetching podcast feed: {url}\n")

    podcast_name, episodes = parse_feed(url)
    print(f"Podcast: {podcast_name}")
    print(f"Found {len(episodes)} episodes\n")

    if not episodes:
        print("No episodes found in feed.", file=sys.stderr)
        sys.exit(1)

    # Filter by search query
    if args.search:
        query = args.search.lower()
        episodes = [
            ep for ep in episodes
            if query in ep.title.lower() or query in ep.description.lower()
        ]
        print(f'Found {len(episodes)} episodes matching "{args.search}":\n')
        if not episodes:
            print("No episodes match your search.", file=sys.stderr)
            sys.exit(1)

    # Display episode list
    display_limit = len(episodes) if args.search else 30
    for i, ep in enumerate(episodes[:display_limit]):
        try:
            dt = datetime.strptime(ep.publish_date[:16], "%a, %d %b %Y")
            date_str = f"{dt.month}/{dt.day}/{dt.year}"
        except (ValueError, IndexError):
            date_str = ep.publish_date[:20] if ep.publish_date else "Unknown date"
        dur = ep.duration or "Unknown duration"
        print(f"  {i + 1:3}. {ep.title}")
        print(f"       {date_str} | {dur}")

    if len(episodes) > display_limit:
        print(f"\n  ... and {len(episodes) - display_limit} more episodes (use --search to filter)")

    # If --list or (--search without --episode), stop here
    if args.list_episodes or (args.search and args.episode is None and not apple_episode_id):
        return

    require_api_key()

    # Select episode
    if apple_episode_id:
        # Try to match the specific episode from the Apple Podcasts link
        apple_info = scrape_apple_episode_info(original_url)
        selected = None
        if apple_info and apple_info.get("title"):
            ep_title = apple_info["title"]
            print(f"\nLooking for episode: {ep_title}")
            for ep in episodes:
                if ep.title.strip().lower() == ep_title.strip().lower():
                    selected = ep
                    break
        if selected is None and apple_info and apple_info.get("audio_url"):
            # Episode not in RSS feed (too old), but we have the audio URL from Apple
            title = apple_info.get("title", "Unknown Episode")
            print(f"\nEpisode not in RSS feed, using audio URL from Apple Podcasts page.")
            selected = Episode(
                title=title,
                audio_url=apple_info["audio_url"],
                publish_date="",
                description="",
                duration="",
            )
        if selected is None:
            print(f"\nCould not find the specific episode. Using most recent episode.")
            selected = episodes[0]
    elif args.episode is not None:
        if args.episode < 1 or args.episode > len(episodes):
            print(f"\nInvalid episode number. Choose between 1 and {len(episodes)}.", file=sys.stderr)
            sys.exit(1)
        selected = episodes[args.episode - 1]
    else:
        # Default: latest
        selected = episodes[0]
        print("\nUsing most recent episode.")

    print(f"\nSelected: {selected.title}")
    print(f"Audio URL: {selected.audio_url}\n")
    print("Starting transcription...")
    print("This may take several minutes depending on episode length.\n")

    t0 = time.time()
    segments, duration = transcribe(selected.audio_url)
    elapsed = int(time.time() - t0)

    print(f"\nTranscription complete in {elapsed} seconds.")
    print(f"Duration: {format_timestamp(duration)}")
    print(f"Segments: {len(segments)}")

    md = generate_markdown(selected.title, podcast_name, segments, duration)
    filename = args.output or f"{sanitize_filename(selected.title)}.md"
    Path(filename).write_text(md, encoding="utf-8")
    print(f"\nSaved to: {filename}")


if __name__ == "__main__":
    main()

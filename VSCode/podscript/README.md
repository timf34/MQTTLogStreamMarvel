# Podscript

CLI tool to transcribe podcasts and YouTube videos using the ElevenLabs Scribe API.

## Installation

```bash
cd podscript
pip install -e .
```

## Setup

Copy `.env.example` to `.env` and add your ElevenLabs API key:

```
ELEVENLABS_API_KEY=your_key_here
```

For YouTube support, install [yt-dlp](https://github.com/yt-dlp/yt-dlp) and [ffmpeg](https://ffmpeg.org/):

```bash
pip install yt-dlp
# ffmpeg: install via your system package manager or https://ffmpeg.org/download.html
```

## Usage

```bash
# List episodes from an RSS feed
podscript https://feeds.simplecast.com/JGE3yC0V --list

# Transcribe the most recent episode
podscript https://feeds.simplecast.com/JGE3yC0V --latest

# Transcribe a specific episode (by number from list)
podscript https://feeds.simplecast.com/JGE3yC0V --episode 3

# Search episodes by keyword
podscript https://feeds.simplecast.com/JGE3yC0V --search "AI"

# Search + select
podscript https://feeds.simplecast.com/JGE3yC0V --search "AI" --episode 2

# Output to a specific file
podscript https://feeds.simplecast.com/JGE3yC0V --latest --output transcript.md

# Transcribe a YouTube video
podscript https://www.youtube.com/watch?v=VIDEO_ID
podscript https://youtu.be/VIDEO_ID --output transcript.md
```

Without any flags, the default behavior is to transcribe the most recent episode.

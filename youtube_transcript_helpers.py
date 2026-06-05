from urllib.parse import parse_qs, urlparse

from youtube_transcript_api import YouTubeTranscriptApi


def extract_video_id(video_url_or_id: str) -> str:
    value = video_url_or_id.strip()
    if not value:
        raise ValueError("Missing YouTube video URL or video ID.")

    if "://" not in value and "/" not in value and "?" not in value:
        return value

    parsed = urlparse(value)
    hostname = parsed.netloc.lower()

    if hostname in {"youtu.be", "www.youtu.be"}:
        video_id = parsed.path.lstrip("/").split("/")[0]
        if video_id:
            return video_id

    if hostname.endswith("youtube.com"):
        query_id = parse_qs(parsed.query).get("v")
        if query_id and query_id[0]:
            return query_id[0]

        path_parts = [part for part in parsed.path.split("/") if part]
        if len(path_parts) >= 2 and path_parts[0] in {"embed", "live", "shorts"}:
            return path_parts[1]

    raise ValueError(f"Could not extract a YouTube video ID from: {video_url_or_id}")


def fetch_transcript_entries(
    video_url_or_id: str,
    languages: tuple[str, ...] = ("en", "en-US"),
    preserve_formatting: bool = False,
) -> list[dict]:
    video_id = extract_video_id(video_url_or_id)
    language_list = list(languages)

    if hasattr(YouTubeTranscriptApi, "get_transcript"):
        return YouTubeTranscriptApi.get_transcript(
            video_id,
            languages=language_list,
            preserve_formatting=preserve_formatting,
        )

    transcript = YouTubeTranscriptApi().fetch(
        video_id,
        languages=language_list,
        preserve_formatting=preserve_formatting,
    )

    if hasattr(transcript, "to_raw_data"):
        return transcript.to_raw_data()

    return [
        {
            "text": item.text,
            "start": item.start,
            "duration": item.duration,
        }
        for item in transcript
    ]


def fetch_transcript_text(
    video_url_or_id: str,
    languages: tuple[str, ...] = ("en", "en-US"),
    preserve_formatting: bool = False,
) -> str:
    entries = fetch_transcript_entries(
        video_url_or_id,
        languages=languages,
        preserve_formatting=preserve_formatting,
    )
    return " ".join(item["text"].strip() for item in entries if item.get("text"))

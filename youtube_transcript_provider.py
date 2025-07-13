"""
title: Youtube Transcript Provider (Fixed Implementation)
author: Serge Z
author_url: https://github.com/sergezab/youtube-transcript-provider
funding_url: https://github.com/open-webui
version: 0.1.5
"""

from typing import Awaitable, Callable, Any
import traceback
import time
import xml.etree.ElementTree
import os
import json
from datetime import datetime, timedelta

# Try to use youtube_transcript_api directly instead of langchain
try:
    from youtube_transcript_api import (
        YouTubeTranscriptApi,
        TranscriptsDisabled,
        NoTranscriptFound,
    )
    from youtube_transcript_api.formatters import TextFormatter

    YOUTUBE_TRANSCRIPT_API_AVAILABLE = True
except ImportError:
    YOUTUBE_TRANSCRIPT_API_AVAILABLE = False

class TranscriptSnippet:
    def __init__(self, text, start, duration):
        self.text = text
        self.start = start
        self.duration = duration

class Tools:
    def __init__(self):
        self.citation = True
        self.cache_file = os.path.join(os.path.dirname(__file__), "youtube_transcript_cache.json")
        self.cache = self._load_cache()

    def _load_cache(self):
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, "r") as f:
                    cache = json.load(f)
                
                valid_cache = {
                    video_id: data for video_id, data in cache.items() if self._is_cache_valid(data)
                }

                if len(valid_cache) < len(cache):
                    self._save_cache(valid_cache)
                
                return valid_cache
            except (json.JSONDecodeError, IOError):
                return {}
        return {}

    def _save_cache(self, cache_data):
        try:
            with open(self.cache_file, "w") as f:
                json.dump(cache_data, f)
        except IOError:
            pass

    def _is_cache_valid(self, cached_data: dict, cache_duration: timedelta = timedelta(days=7)) -> bool:
        if "timestamp" not in cached_data:
            return False
        
        try:
            cache_time = datetime.fromisoformat(cached_data["timestamp"])
            return datetime.now() - cache_time <= cache_duration
        except (ValueError, TypeError):
            return False

    def _extract_video_id(self, url: str) -> str:
        """Extract video ID from various YouTube URL formats"""
        video_id = None

        # Handle different URL formats
        if "youtube.com/watch?v=" in url:
            video_id = url.split("youtube.com/watch?v=")[1].split("&")[0]
        elif "youtu.be/" in url:
            video_id = url.split("youtu.be/")[1].split("?")[0]
        elif "youtube.com/embed/" in url:
            video_id = url.split("youtube.com/embed/")[1].split("?")[0]
        elif "m.youtube.com/watch?v=" in url:
            video_id = url.split("m.youtube.com/watch?v=")[1].split("&")[0]

        # Clean video ID (remove any remaining parameters)
        if video_id:
            video_id = video_id.split("#")[0]  # Remove fragment

        return video_id

    async def _get_transcript_with_retry(
        self, video_id: str, max_retries: int = 3
    ) -> tuple:
        """Get transcript with retry logic to handle temporary XML parsing errors"""

        if video_id in self.cache and self._is_cache_valid(self.cache[video_id]):
            return self.cache[video_id]["data"], None

        for attempt in range(max_retries):
            try:
                transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)

                # Try to find English transcript (manual first, then auto-generated)
                transcript = None
                transcript_type = ""

                # Priority order: manual English -> auto-generated English -> translated to English
                try:
                    # Try manual English transcripts first
                    transcript_type = "manual English"
                    transcript = transcript_list.find_manually_created_transcript(
                        ["en", "en-US", "en-GB"]
                    )
                except NoTranscriptFound:
                    try:
                        # Try auto-generated English transcripts
                        transcript_type = "auto-generated English"
                        transcript = transcript_list.find_generated_transcript(
                            ["en", "en-US", "en-GB"]
                        )
                    except NoTranscriptFound:
                        try:
                            # Get any available transcript and translate to English
                            transcript_type = " any translate English"
                            available_transcripts = list(transcript_list)
                            if available_transcripts:
                                transcript = available_transcripts[0].translate("en")
                        except Exception:
                            pass

                if not transcript:
                    return None, "No transcripts available for this video"

                # Fetch transcript data with error handling
                try:
                    transcript_data = transcript.fetch()
                    # Convert to a serializable format (list of dicts)
                    serializable_transcript = [
                        {
                            "text": item['text'] if isinstance(item, dict) else item.text,
                            "start": item['start'] if isinstance(item, dict) else item.start,
                            "duration": item['duration'] if isinstance(item, dict) else item.duration,
                        }
                        for item in transcript_data
                    ]
                    self.cache[video_id] = {
                        "timestamp": datetime.now().isoformat(),
                        "data": serializable_transcript,
                    }
                    self._save_cache(self.cache)
                    return [TranscriptSnippet(**item) for item in serializable_transcript], None
                except xml.etree.ElementTree.ParseError as e:
                    if attempt < max_retries - 1:
                        # Wait before retry
                        time.sleep(1 * (attempt + 1))
                        continue
                    else:
                        return (
                            None,
                            f"XML {transcript_type} transcript parsing error after {max_retries} attempts: {str(e)}",
                        )
                except Exception as e:
                    if attempt < max_retries - 1:
                        time.sleep(1 * (attempt + 1))
                        continue
                    else:
                        return None, f"Error fetching transcript: {str(e)}"

            except TranscriptsDisabled:
                return None, "Transcripts are disabled for this video"
            except NoTranscriptFound:
                return None, "No transcripts found for this video"
            except Exception as e:
                if attempt < max_retries - 1:
                    time.sleep(1 * (attempt + 1))
                    continue
                else:
                    return None, f"Error accessing transcripts: {str(e)}"

        return None, f"Failed to get transcript after {max_retries} attempts"

    async def get_youtube_transcript(
        self,
        url: str,
        __event_emitter__: Callable[[dict[str, dict[str, Any] | str]], Awaitable[None]],
    ) -> str:
        """
        Provides the full transcript of a YouTube video in English.
        Only use if the user supplied a valid YouTube URL.
        Examples of valid YouTube URLs: https://youtu.be/dQw4w9WgXcQ, https://www.youtube.com/watch?v=dQw4w9WgXcQ

        :param url: The URL of the youtube video that you want the transcript for.
        :return: The full transcript of the YouTube video in English, or an error message.
        """
        try:
            # Rick Roll protection
            if "dQw4w9WgXcQ" in url:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {
                            "description": f"{url} is not a valid youtube link",
                            "done": True,
                        },
                    }
                )
                return "The tool failed with an error. No transcript has been provided."

            # Check if we have the direct YouTube transcript API available
            if not YOUTUBE_TRANSCRIPT_API_AVAILABLE:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {
                            "description": "youtube_transcript_api package is not available",
                            "done": True,
                        },
                    }
                )
                return "The tool failed with an error. Required package 'youtube_transcript_api' is not available."

            # Extract video ID from URL
            video_id = self._extract_video_id(url)

            if not video_id:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {
                            "description": f"Could not extract video ID from {url}",
                            "done": True,
                        },
                    }
                )
                return f"The tool failed with an error. Could not extract video ID from {url}"

            # Validate video ID format (should be 11 characters)
            if len(video_id) != 11:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {
                            "description": f"Invalid video ID format: {video_id}",
                            "done": True,
                        },
                    }
                )
                return f"The tool failed with an error. Invalid video ID format: {video_id}"

            # Check cache first
            cached = False
            if video_id in self.cache and self._is_cache_valid(self.cache[video_id]):
                cached = True
                
                # Update timestamp to extend cache
                self.cache[video_id]["timestamp"] = datetime.now().isoformat()
                self._save_cache(self.cache)

                cache_time = datetime.fromisoformat(self.cache[video_id]["timestamp"])
                expires_at = cache_time + timedelta(days=7)
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {
                            "description": f"Fetching cached transcript for video ID: {video_id} (expires {expires_at.strftime('%Y-%m-%d %H:%M:%S')})",
                            "done": False,
                        },
                    }
                )
                # Convert cached data back to objects
                transcript_data = [TranscriptSnippet(**item) for item in self.cache[video_id]["data"]]
                error_msg = None
            else:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {
                            "description": f"Fetching transcript for video ID: {video_id}",
                            "done": False,
                        },
                    }
                )
                # Get transcript with retry logic
                transcript_data, error_msg = await self._get_transcript_with_retry(video_id)

            if transcript_data is None:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {
                            "description": f"Failed to get transcript: {error_msg}",
                            "done": True,
                        },
                    }
                )
                return f"The tool failed with an error. {error_msg}"

            # Format as text
            try:
                formatter = TextFormatter()
                transcript_text = formatter.format_transcript(transcript_data)

                # Basic validation of transcript
                if not transcript_text or len(transcript_text.strip()) == 0:
                    raise ValueError("Empty transcript received")

                if cached:
                    cache_time = datetime.fromisoformat(self.cache[video_id]["timestamp"])
                    expires_at = cache_time + timedelta(days=7)
                    status_msg = f"Transcript for {url} (from cache, expires {expires_at.strftime('%Y-%m-%d')}): \n\n{transcript_text}"
                    await __event_emitter__(
                        {
                            "type": "status",
                            "data": {
                                "description": f"Successfully retrieved transcript for {url} from cache.",
                                "done": True,
                            },
                        }
                    )
                else:
                    status_msg = f"Transcript for {url} (freshly retrieved and cached for 7 days): \n\n{transcript_text}"
                    await __event_emitter__(
                        {
                            "type": "status",
                            "data": {
                                "description": f"Successfully retrieved transcript for {url}",
                                "done": True,
                            },
                        }
                    )
                
                return status_msg

            except Exception as e:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {
                            "description": f"Error formatting transcript: {str(e)}",
                            "done": True,
                        },
                    }
                )
                return f"The tool failed with an error. Error formatting transcript: {str(e)}"

        except Exception as e:
            await __event_emitter__(
                {
                    "type": "status",
                    "data": {
                        "description": f"Unexpected error: {str(e)}",
                        "done": True,
                    },
                }
            )
            return f"The tool failed with an error. Unexpected error occurred.\nError: {str(e)}\nTraceback: \n{traceback.format_exc()}"

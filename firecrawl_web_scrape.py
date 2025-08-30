"""
title: Firecrawl Web Scrape (with Progress)
description: A simplified, robust web scraping tool using the Firecrawl V2 API, with real-time progress updates.
author: Artur Zdolinski (Refactored by Gemini)
author_url: https://github.com/azdolinski
git_url: https://github.com/azdolinski/open-webui-tools
required_open_webui_version: 0.4.0
requirements: pydantic, aiohttp, beautifulsoup4, html2text
version: 1.0.2 [2025-08-29]
licence: MIT
"""

import json
import logging
import asyncio
import os
import shelve
import re
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Dict, List
from functools import wraps

import aiohttp
from pydantic import BaseModel, Field
from bs4 import BeautifulSoup
import html2text

# region: --- Logging Configuration ---
# ==============================================================================
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
# ==============================================================================
# endregion


# region: --- EventEmitter and Decorators ---
# ==============================================================================
class EventEmitter:
    """A helper class to emit status updates."""

    def __init__(self, event_emitter: Callable[[dict], Any] = None):
        self.event_emitter = event_emitter

    async def progress_update(self, description: str):
        await self.emit(description=description, status="in_progress")

    async def error_update(self, description: str):
        await self.emit(description=description, status="error", done=True)

    async def success_update(self, description: str):
        await self.emit(description=description, status="success", done=True)

    async def emit(self, description="Unknown State", status="in_progress", done=False):
        if self.event_emitter:
            await self.event_emitter(
                {
                    "type": "status",
                    "data": {
                        "description": description,
                        "status": status,
                        "done": done,
                    },
                }
            )


def async_retry_with_backoff(max_retries=3, initial_backoff=2, max_backoff=30):
    """
    A decorator to retry an async function with exponential backoff.
    """

    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            retry_count = 0
            backoff = initial_backoff
            while retry_count < max_retries:
                try:
                    return await func(*args, **kwargs)
                except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                    logger.warning(
                        f"Request failed: {e}. Retrying ({retry_count + 1}/{max_retries})..."
                    )
                    retry_count += 1
                    if retry_count >= max_retries:
                        logger.error(
                            f"Max retries reached for {func.__name__}. Last error: {e}"
                        )
                        raise
                    await asyncio.sleep(backoff + (0.5 * retry_count))
                    backoff = min(backoff * 2, max_backoff)
            return None

        return wrapper

    return decorator


# ==============================================================================
# endregion


# region: --- Caching Utilities ---
# ==============================================================================
def is_cache_stale(timestamp_str: str, ttl_seconds: int) -> bool:
    """Check if a cached item is older than the specified TTL."""
    if ttl_seconds <= 0:
        return True  # Caching is disabled
    try:
        # Timestamps are stored with timezone info
        cache_time = datetime.fromisoformat(timestamp_str)
        return (datetime.now(timezone.utc) - cache_time) > timedelta(
            seconds=ttl_seconds
        )
    except (ValueError, TypeError):
        return True


# ==============================================================================
# endregion


class Tools:
    class Valves(BaseModel):
        # --- Core User Configuration ---
        firecrawl_api_key: str = ""
        formats: List[str] = Field(
            default=["markdown"],
            description="Output formats. 'markdown' is best for LLMs. Post-processing: 'html2text', 'html2bs4'.",
        )
        request_timeout: int = Field(
            120, description="Timeout for the HTTP request in seconds."
        )
        cache_ttl_seconds: int = Field(
            3600, description="Local cache duration in seconds. Set to 0 to disable."
        )

        # --- Internal Configuration ---
        firecrawl_api_url: str = "https://api.firecrawl.dev/v2/scrape"

    def __init__(self):
        self.valves = self.Valves()
        self._cache_path = os.path.join(os.path.dirname(__file__), "firecrawl_cache.db")
        logger.info(f"Using cache file at: {self._cache_path}")

    # region: --- Helper Methods ---
    # ==========================================================================
    def _get_cache_key(self, url: str, payload: Dict[str, Any]) -> str:
        """Creates a consistent cache key from URL and payload."""
        payload_str = json.dumps(payload, sort_keys=True)
        return f"{url}::{payload_str}"

    def text_cleaner(self, text: str) -> str:
        """Cleans up markdown text."""
        cleaned = re.sub(r"\\+", "", text)
        cleaned = re.sub(r"\[.*?\]\((?!(?:http|mailto:)).*?\)", "", cleaned)
        cleaned = re.sub(r"^[=\-#\*]{3,}$", "", cleaned, flags=re.MULTILINE)
        cleaned = re.sub(r"\n\s*\n", "\n", cleaned)
        return cleaned.strip()

    def html_clean_bs4(self, html_content: str) -> str:
        """Cleans common unwanted HTML tags."""
        soup = BeautifulSoup(html_content, "html.parser")
        for tag in soup.find_all(["script", "style", "head", "iframe", "meta", "svg"]):
            tag.decompose()
        return str(soup)

    def html_clean_html2text(self, html_content: str) -> str:
        """Converts HTML to clean Markdown."""
        h = html2text.HTML2Text(bodywidth=0)
        h.ignore_links = True
        h.ignore_images = True
        return self.text_cleaner(h.handle(html_content))

    # ==========================================================================
    # endregion

    @async_retry_with_backoff()
    async def _make_firecrawl_request(
        self, payload: Dict[str, Any], event_emitter: Any
    ) -> Dict:
        """Performs the async HTTP request to Firecrawl with retry logic and progress updates."""
        headers = {
            "Authorization": f"Bearer {self.valves.firecrawl_api_key}",
            "Content-Type": "application/json",
        }
        timeout = aiohttp.ClientTimeout(total=self.valves.request_timeout)

        progress_task = None

        async def progress_updater():
            """Updates the status every 5 seconds with an elapsed timer."""
            start_time = asyncio.get_event_loop().time()
            while True:
                await asyncio.sleep(5)
                elapsed = int(asyncio.get_event_loop().time() - start_time)
                await event_emitter.progress_update(
                    f"Scraping content... ({elapsed}s elapsed)"
                )

        try:
            if event_emitter:
                progress_task = asyncio.create_task(progress_updater())

            async with aiohttp.ClientSession(
                headers=headers, timeout=timeout
            ) as session:
                logger.debug(
                    f"Making async request to {self.valves.firecrawl_api_url} with payload: {payload}"
                )
                async with session.post(
                    self.valves.firecrawl_api_url, json=payload
                ) as response:
                    response.raise_for_status()
                    return await response.json()
        finally:
            if progress_task:
                progress_task.cancel()
                # Wait for the cancellation to complete to avoid stray errors
                await asyncio.gather(progress_task, return_exceptions=True)

    async def web_scrape(
        self, url: str, __user__: dict = None, __event_emitter__=None
    ) -> str:
        """
        Scrapes a webpage using Firecrawl V2 API, returning its content.

        :param url: The URL to scrape.
        :return: The scraped content as a formatted string.
        """
        event_emitter = EventEmitter(__event_emitter__)
        await event_emitter.progress_update(f"Starting web scrape for {url}...")

        try:
            # --- 1. Prepare Request Payload ---
            if not url.startswith("http"):
                url = f"https://{url}"

            payload = {
                "url": url,
                "onlyMainContent": True,
                "removeBase64Images": True,
                "blockAds": True,
                "storeInCache": True,
                "proxy": "auto",
                "timeout": 120000,
            }
            payload["formats"] = [
                f for f in self.valves.formats if not f.startswith("html2")
            ]

            # --- 2. Check Local Cache ---
            cache_key = self._get_cache_key(url, payload)
            if self.valves.cache_ttl_seconds > 0:
                with shelve.open(self._cache_path) as cache:
                    if cache_key in cache and not is_cache_stale(
                        cache[cache_key]["timestamp"], self.valves.cache_ttl_seconds
                    ):
                        logger.info(f"Returning local cached content for {url}")
                        await event_emitter.success_update(
                            f"Successfully retrieved cached content for {url}"
                        )
                        return cache[cache_key]["data"]

            # --- 3. Make API Call with Progress Updates ---
            full_response = await self._make_firecrawl_request(payload, event_emitter)
            scraped_data = full_response.get("data", {})

            # Check if any usable content key exists
            if not scraped_data or not any(
                key in scraped_data for key in ["markdown", "html", "content"]
            ):
                error_msg = f"Error: Firecrawl response missing expected content keys. Full Response: {full_response}"
                await event_emitter.error_update(error_msg)
                return error_msg

            # --- 4. Process Content ---
            await event_emitter.progress_update("Processing scraped content...")
            content = {}
            for format_type in self.valves.formats:
                if format_type == "markdown":
                    # Fallback to 'content' key if 'markdown' is not present
                    markdown_text = scraped_data.get(
                        "markdown", scraped_data.get("content", "")
                    )
                    content["markdown"] = self.text_cleaner(markdown_text)
                elif format_type == "html":
                    content["html"] = scraped_data.get("html", "")
                elif format_type == "html2text":
                    html_content = scraped_data.get("html", "")
                    content["html2text"] = (
                        self.html_clean_html2text(html_content) if html_content else ""
                    )
                elif format_type == "html2bs4":
                    html_content = scraped_data.get("html", "")
                    content["html2bs4"] = (
                        self.html_clean_bs4(html_content) if html_content else ""
                    )
                else:
                    content[format_type] = scraped_data.get(format_type, "")

            # --- 5. Format and Cache Final Output ---
            metadata = scraped_data.get("metadata", {})
            final_output = (
                f"Date: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}\n"
                f"Source URL: {url}\n"
                f"Title: {metadata.get('title', 'N/A')}\n"
                f"---\n"
                f"{json.dumps(content, indent=2, ensure_ascii=False)}\n"
            )

            if self.valves.cache_ttl_seconds > 0:
                with shelve.open(self._cache_path, "c") as cache:
                    cache[cache_key] = {
                        "data": final_output,
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    }

            await event_emitter.success_update(
                f"Successfully scraped content from {url}"
            )
            return final_output.strip()

        except aiohttp.ClientResponseError as e:
            error_msg = f"API Error ({e.status}): {await e.text()}"
            logger.error(f"HTTP error for {url}: {error_msg}")
            await event_emitter.error_update(error_msg)
            return error_msg
        except Exception as e:
            error_msg = f"An unexpected error occurred: {str(e)}"
            logger.error(f"Exception during web scrape for {url}: {e}", exc_info=True)
            await event_emitter.error_update(error_msg)
            return error_msg

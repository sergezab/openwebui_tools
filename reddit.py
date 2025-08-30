"""
title: Reddit Tool
description: A tool to fetch posts from subreddits, users, and to perform searches across Reddit. Includes caching and content filtering.
author: @nathanwindisch (Refactored by Gemini & Sergii Zabigailo)
version: 1.3.0
"""

import re
import json
import requests
import os
import shelve
import time
from typing import Awaitable, Callable, List
from pydantic import BaseModel, Field
from requests.models import Response
import logging

# --- Logging Configuration ---
logger = logging.getLogger(__name__)
if not logger.handlers:
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    try:
        log_dir = os.path.expanduser("~")
        log_file = os.path.join(log_dir, "reddit_tool.log")
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    except Exception as e:
        logger.error(f"Failed to set up file logger: {e}")

# --- Caching and Time Utilities ---


def is_cache_stale(cache_entry: dict, ttl_seconds: int = 1800) -> bool:  # 30 minute TTL
    if "timestamp" not in cache_entry:
        return True
    return (time.time() - cache_entry["timestamp"]) > ttl_seconds


def get_relative_time(utc_timestamp: float) -> str:
    if not utc_timestamp:
        return ""
    now = time.time()
    diff = now - utc_timestamp

    if diff < 60:
        return "just now"
    elif diff < 3600:
        return f"{int(diff / 60)} minutes ago"
    elif diff < 86400:
        return f"{int(diff / 3600)} hours ago"
    elif diff < 2592000:
        return f"{int(diff / 86400)} days ago"
    elif diff < 31536000:
        return f"{int(diff / 2592000)} months ago"
    else:
        return f"{int(diff / 31536000)} years ago"


# --- Data Parsing Functions ---


def parse_reddit_page(response: Response) -> list:
    try:
        data = response.json()
        return data.get("data", {}).get("children", [])
    except json.JSONDecodeError:
        logger.error("Failed to decode JSON from Reddit response.")
        return []


def parse_posts(data: list) -> list:
    """
    Parse a list of post items into a structured format, with robust filtering.
    """
    posts = []
    for item in data:
        if item.get("kind") != "t3":
            continue
        item_data = item.get("data", {})

        author = item_data.get("author", "")
        if author.lower() == "automoderator":
            continue

        score = item_data.get("score", 0)
        total_comments = item_data.get("num_comments", 0)
        published_at = item_data.get("created_utc")

        is_new = (time.time() - published_at) < 3600 if published_at else False
        if not is_new and (score < 3 and total_comments < 2):
            continue

        domain = item_data.get("domain", "").lower()
        url = item_data.get("url", "").lower()

        if domain in ["i.redd.it", "youtube.com", "youtu.be", "v.redd.it"]:
            continue
        if "reddit.com/gallery/" in url:
            continue

        if url.endswith((".png", ".jpg", ".jpeg", ".gif", ".webp")):
            continue

        posts.append(
            {
                "id": item_data.get("name"),
                "title": item_data.get("title"),
                "description": item_data.get("selftext"),
                "link": item_data.get("url"),
                "author_username": item_data.get("author"),
                "author_id": item_data.get("author_fullname"),
                "subreddit_name": item_data.get("subreddit"),
                "subreddit_id": item_data.get("subreddit_id"),
                "score": item_data.get("score", 0),
                "total_comments": item_data.get("num_comments", 0),
                "is_self": item_data.get("is_self", False),
                "published_at": item_data.get("created_utc"),
            }
        )
    return posts


def parse_comments(data: list) -> list:
    comments = []
    for item in data:
        if item.get("kind") != "t1":
            continue
        item_data = item.get("data", {})
        comments.append(
            {
                "id": item_data.get("name"),
                "body": item_data.get("body"),
                "link": item_data.get("permalink"),
                "post_id": item_data.get("link_id"),
                "author_username": item_data.get("author"),
                "score": item_data.get("score", 0),
                "published_at": item_data.get("created_utc"),
            }
        )
    return comments


# --- Formatting and Summarization Functions ---


def summarize_text(text: str, max_length: int = 400) -> str:
    if not text or len(text) <= max_length:
        return text

    sentences = re.split(r"(?<=[.!?])\s+", text)
    if not sentences:
        return text[:max_length]

    summary = sentences[0]
    for sentence in sentences[1:-1]:
        if len(summary) + len(sentence) + 5 < max_length:
            summary += " " + sentence
        else:
            break

    if len(sentences) > 1 and len(summary) + len(sentences[-1]) + 5 < max_length:
        summary += " [...] " + sentences[-1]
    else:
        summary += "..."
    return summary.strip()


def format_posts_to_human_readable(posts: list) -> str:
    if not posts:
        return "No relevant posts found."
    output = []
    for post in posts:
        raw_description = (post.get("description", "") or "").strip()
        description_no_urls = re.sub(r"https?://\S+", "", raw_description).strip()

        description_part = ""
        if description_no_urls:
            summary = summarize_text(description_no_urls, 800)
            description_part = f"\nDescription: {summary}"

        post_age = get_relative_time(post.get("published_at"))

        output.append(
            f"Title: {post.get('title', 'N/A')}\n"
            f"Subreddit: r/{post.get('subreddit_name', 'N/A')} ({post_age})\n"
            f"Author: u/{post.get('author_username', 'N/A')}\n"
            f"Score: {post.get('score', 0)} | Comments: {post.get('total_comments', 0)}\n"
            f"Link: {post.get('link', '#')}"
            f"{description_part}"
        )
    return "\n---\n".join(output)


def format_comments_to_human_readable(comments: list) -> str:
    if not comments:
        return "No comments found."
    output = []
    for comment in comments:
        raw_body = (comment.get("body", "") or "").strip()
        body_no_urls = re.sub(r"https?://\S+", "", raw_body).strip()
        summary = summarize_text(body_no_urls, 450)
        comment_age = get_relative_time(comment.get("published_at"))

        output.append(
            f"Author: u/{comment.get('author_username', 'N/A')} | Score: {comment.get('score', 0)} ({comment_age})\n"
            f"Comment: {summary}\n"
            f"Link: https://reddit.com{comment.get('link', '#')}"
        )
    return "\n---\n".join(output)


# --- Main Tool Class ---


class Tools:
    class UserValves(BaseModel):
        USER_AGENT: str = Field(
            default="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36",
            description="The user agent to use when making requests to Reddit.",
        )

    def __init__(self):
        self.tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_subreddit_feed",
                    "description": "Get the latest posts from a specific subreddit.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "subreddit": {
                                "type": "string",
                                "description": "The name of the subreddit (e.g., 'stocks').",
                            }
                        },
                        "required": ["subreddit"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "get_user_feed",
                    "description": "Get the latest posts and comments from a specific Reddit user. Use a real Reddit username.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "username": {
                                "type": "string",
                                "description": "The Reddit username (e.g., 'gregoleg').",
                            }
                        },
                        "required": ["username"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "search_reddit",
                    "description": "Search for relevant posts containing specific keywords, either across all of Reddit or within a specific subreddit.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "The search term or keywords (e.g., 'AAPL').",
                            },
                            "subreddit": {
                                "type": "string",
                                "description": "Optional. The subreddit to search within. Defaults to a relevant financial subreddit for stock-related queries.",
                                "default": "all",
                            },
                        },
                        "required": ["query"],
                    },
                },
            },
        ]
        self.cache_path = os.path.join(os.path.dirname(__file__), "reddit_cache.db")
        logger.info("Reddit Tool initialized.")

    async def _make_request(
        self, url: str, headers: dict, cache_key: str, __event_emitter__
    ):
        with shelve.open(self.cache_path) as cache:
            if cache_key in cache and not is_cache_stale(cache[cache_key]):
                if __event_emitter__:
                    await __event_emitter__(
                        {
                            "type": "status",
                            "data": {
                                "description": f"Loading '{cache_key}' from cache.",
                                "done": False,
                            },
                        }
                    )
                logger.info(f"Cache hit for key: {cache_key}")
                return cache[cache_key]["data"]

            try:
                logger.info(f"Cache miss. Making request to URL: {url}")
                response = requests.get(url, headers=headers)
                if not response.ok:
                    logger.error(
                        f"Request failed with status {response.status_code} for URL: {url}"
                    )
                    return f"Error: {response.status_code}"

                parsed_data = parse_reddit_page(response)
                logger.info(
                    f"Successfully fetched and parsed data for key: {cache_key}"
                )
                cache[cache_key] = {"data": parsed_data, "timestamp": time.time()}
                return parsed_data
            except Exception as e:
                logger.error(
                    f"An exception occurred during request: {e}", exc_info=True
                )
                return f"Error: {e}"

    async def get_subreddit_feed(
        self, subreddit: str, __user__={}, __event_emitter__=None
    ) -> str:
        if not subreddit:
            logger.warning("get_subreddit_feed called with no subreddit.")
            return "Error: Subreddit name not provided."
        subreddit = subreddit.replace("/r/", "").replace("r/", "")
        url = f"https://reddit.com/r/{subreddit}.json"
        cache_key = f"subreddit_{subreddit}"
        headers = {"User-Agent": __user__.get("valves", self.UserValves()).USER_AGENT}

        if __event_emitter__:
            await __event_emitter__(
                {
                    "type": "status",
                    "data": {
                        "description": f"Fetching feed for r/{subreddit}...",
                        "done": False,
                    },
                }
            )
        result = await self._make_request(url, headers, cache_key, __event_emitter__)

        if isinstance(result, str):
            if __event_emitter__:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {
                            "description": f"Failed to get feed for r/{subreddit}.",
                            "done": True,
                        },
                    }
                )
            return result

        posts = parse_posts(result)
        if __event_emitter__:
            await __event_emitter__(
                {
                    "type": "status",
                    "data": {
                        "description": f"Feed for r/{subreddit} retrieved.",
                        "done": True,
                    },
                }
            )
        return format_posts_to_human_readable(posts)

    async def get_user_feed(
        self, username: str, __user__={}, __event_emitter__=None
    ) -> str:
        if not username:
            logger.warning("get_user_feed called with no username.")
            return "Error: Username not provided."
        username = username.replace("/u/", "").replace("u/", "")
        url = f"https://reddit.com/u/{username}.json"
        cache_key = f"user_{username}"
        headers = {"User-Agent": __user__.get("valves", self.UserValves()).USER_AGENT}

        if __event_emitter__:
            await __event_emitter__(
                {
                    "type": "status",
                    "data": {
                        "description": f"Fetching feed for u/{username}...",
                        "done": False,
                    },
                }
            )
        result = await self._make_request(url, headers, cache_key, __event_emitter__)

        if isinstance(result, str):
            if __event_emitter__:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {
                            "description": f"Failed to get feed for u/{username}.",
                            "done": True,
                        },
                    }
                )
            return result

        posts = parse_posts(result)
        comments = parse_comments(result)

        formatted_posts = (
            f"--- POSTS ---\n{format_posts_to_human_readable(posts)}" if posts else ""
        )
        formatted_comments = (
            f"\n\n--- COMMENTS ---\n{format_comments_to_human_readable(comments)}"
            if comments
            else ""
        )

        if __event_emitter__:
            await __event_emitter__(
                {
                    "type": "status",
                    "data": {
                        "description": f"Feed for u/{username} retrieved.",
                        "done": True,
                    },
                }
            )
        if not formatted_posts and not formatted_comments:
            return "No activity found for this user."

        return f"{formatted_posts}{formatted_comments}".strip()

    async def search_reddit(
        self, query: str, subreddit: str = "all", __user__={}, __event_emitter__=None
    ) -> str:
        # Use the User-Agent from valves for consistency
        headers = {"User-Agent": __user__.get("valves", self.UserValves()).USER_AGENT}

        query_subreddit_match = re.match(r"r/(\w+)\s+(.*)", query)
        if query_subreddit_match:
            subreddit = query_subreddit_match.group(1)
            query = query_subreddit_match.group(2)

        # Return to sorting by new for the most recent results
        sort_param = "&sort=new"

        GENERAL_FINANCE_KEYWORDS = [
            "market",
            "stocks",
            "investing",
            "trade",
            "economy",
            "financial",
        ]
        is_general_finance_query = any(
            keyword in query.lower() for keyword in GENERAL_FINANCE_KEYWORDS
        )

        subreddit_for_cache = subreddit
        if subreddit.lower() in ["r", "all", ""] and is_general_finance_query:
            subreddit_name = "stocks"
            subreddit_for_cache = subreddit_name
            url = f"https://www.reddit.com/r/{subreddit_name}/search.json?q={query}&restrict_sr=on{sort_param}"
            search_scope = f"r/{subreddit_name} (auto-selected for relevance)"
            logger.info(
                f"General finance query detected. Focusing search on {search_scope}."
            )
        elif subreddit.lower() in ["r", "all", ""]:
            subreddit_for_cache = "all"
            url = f"https://www.reddit.com/search.json?q={query}{sort_param}"
            search_scope = "all of Reddit"
        else:
            subreddit_name = subreddit.replace("/r/", "").replace("r/", "")
            subreddit_for_cache = subreddit_name
            url = f"https://www.reddit.com/r/{subreddit_name}/search.json?q={query}&restrict_sr=on{sort_param}"
            search_scope = f"r/{subreddit_name}"

        # Incremented cache key to ensure fresh, newest-sorted results.
        cache_key = f"v10_search_{subreddit_for_cache}_{query}"

        if __event_emitter__:
            await __event_emitter__(
                {
                    "type": "status",
                    "data": {
                        "description": f"Searching for '{query}' in {search_scope}...",
                        "done": False,
                    },
                }
            )
        result = await self._make_request(url, headers, cache_key, __event_emitter__)

        if isinstance(result, str):
            if __event_emitter__:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {
                            "description": f"Search for '{query}' failed.",
                            "done": True,
                        },
                    }
                )
            return result

        posts = parse_posts(result)
        if __event_emitter__:
            await __event_emitter__(
                {
                    "type": "status",
                    "data": {
                        "description": f"Search for '{query}' complete.",
                        "done": True,
                    },
                }
            )
        return format_posts_to_human_readable(posts)

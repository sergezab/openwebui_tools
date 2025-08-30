"""
title: Reddit
author: @nathanwindisch
author_url: https://git.wnd.sh/owui-tools/reddit
funding_url: https://patreon.com/NathanWindisch
version: 0.0.5
changelog:
- 0.0.1 - Initial upload to openwebui community.
- 0.0.2 - Renamed from "Reddit Feeds" to just "Reddit".
- 0.0.3 - Updated author_url in docstring to point to git repo.
- 0.0.4 - Updated to return human-readable output.
- 0.0.5 - Refactored to a more direct tool structure and added caching.
"""

import re
import json
import requests
import os
import shelve
import time
from typing import Awaitable, Callable, Dict, Any
from pydantic import BaseModel, Field
from requests.models import Response

# --- Caching Configuration ---
CACHE_DURATION = 3600  # Cache duration in seconds (1 hour)

def is_cache_stale(timestamp: float) -> bool:
    """Check if the cache is older than the CACHE_DURATION."""
    return (time.time() - timestamp) > CACHE_DURATION

def update_cache(cache: shelve.Shelf, key: str, data: Any):
    """Updates the cache for a given key with a timestamp."""
    cache[key] = {
        "data": data,
        "timestamp": time.time(),
    }

# --- Data Parsing Functions ---

def parse_reddit_page(response: Response):
    data = json.loads(response.content)
    output = []
    if "data" not in data:
        return output
    if "children" not in data["data"]:
        return output
    for item in data["data"]["children"]:
        output.append(item)
    return output


def parse_posts(data: list):
    posts = []
    for item in data:
        if item["kind"] != "t3":
            continue
        item = item["data"]
        posts.append(
            {
                "id": item["name"],
                "title": item["title"],
                "description": item["selftext"],
                "link": item["url"],
                "author_username": item["author"],
                "author_id": item["author_fullname"],
                "subreddit_name": item["subreddit"],
                "subreddit_id": item["subreddit_id"],
                "subreddit_subscribers": item["subreddit_subscribers"],
                "score": item["score"],
                "upvotes": item["ups"],
                "downvotes": item["downs"],
                "upvote_ratio": item["upvote_ratio"],
                "total_comments": item["num_comments"],
                "total_crossposts": item["num_crossposts"],
                "total_awards": item["total_awards_received"],
                "domain": item["domain"],
                "flair_text": item["link_flair_text"],
                "media_embed": item.get("media_embed", {}),
                "is_pinned": item.get("pinned", False),
                "is_self": item.get("is_self", False),
                "is_video": item.get("is_video", False),
                "is_media_only": item.get("is_media_only", False),
                "is_over_18": item.get("over_18", False),
                "is_edited": item.get("edited", False),
                "is_hidden": item.get("hidden", False),
                "is_archived": item.get("archived", False),
                "is_locked": item.get("locked", False),
                "is_quarantined": item.get("quarantine", False),
                "is_spoiler": item.get("spoiler", False),
                "is_stickied": item.get("stickied", False),
                "is_send_replies": item.get("send_replies", False),
                "published_at": item.get("created_utc"),
            }
        )
    return posts

# --- Formatting Function ---

def format_posts_to_human_readable(posts: list) -> str:
    """Formats a list of Reddit posts into a human-readable string."""
    if not posts:
        return "No posts found."

    output = []
    for post in posts:
        description = post['description']
        if len(description) > 200:
            description = description[:200] + "..."

        output.append(
            f"Title: {post['title']}\n"
            f"Subreddit: r/{post['subreddit_name']}\n"
            f"Author: u/{post['author_username']}\n"
            f"Score: {post['score']} (Upvotes: {post['upvotes']}, Downvotes: {post['downvotes']})\n"
            f"Comments: {post['total_comments']}\n"
            f"Link: {post['link']}\n"
            f"Description: {description}\n"
        )
    return "\n---\n\n".join(output)

# --- Main Tool Class ---

class Tools:
    def __init__(self):
        self.citation = True
        self.tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_subreddit_feed",
                    "description": "Retrieves the latest posts from a subreddit feed.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "subreddit": {
                                "type": "string",
                                "description": "The name of the subreddit to retrieve posts from.",
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
                    "description": "Retrieves the latest posts from a user's feed.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "username": {
                                "type": "string",
                                "description": "The username of the user to retrieve posts from.",
                            }
                        },
                        "required": ["username"],
                    },
                },
            },
        ]
        pass

    async def get_subreddit_feed(
        self,
        subreddit: str,
        __event_emitter__: Callable[[dict], Awaitable[None]],
    ) -> str:
        """
        Retrieves the latest posts from a subreddit feed.
        """
        cache_path = os.path.join(os.path.dirname(__file__), "reddit_cache.db")
        with shelve.open(cache_path) as cache:
            cache_key = f"subreddit_{subreddit}"
            cached_item = cache.get(cache_key)

            if cached_item and not is_cache_stale(cached_item["timestamp"]):
                await __event_emitter__({
                    "data": {"description": f"Returning cached posts for r/{subreddit}...", "status": "complete", "done": True},
                    "type": "status"
                })
                return cached_item["data"]

            try:
                await __event_emitter__({
                    "data": {"description": f"Retrieving latest posts from r/{subreddit}...", "status": "running"},
                    "type": "status"
                })
                response = requests.get(
                    f"https://www.reddit.com/r/{subreddit}.json",
                    headers={
                        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
                    },
                )
                if response.status_code != 200:
                    await __event_emitter__({
                        "data": {"description": f"Failed to retrieve any posts from r/{subreddit}: {response.status_code}.", "status": "complete", "done": True},
                        "type": "status"
                    })
                    return f"Error: {response.status_code}"
                else:
                    posts = parse_posts(parse_reddit_page(response))
                    formatted_posts = format_posts_to_human_readable(posts)
                    update_cache(cache, cache_key, formatted_posts)
                    
                    await __event_emitter__({
                        "data": {"description": f"Retrieved {len(posts)} posts from r/{subreddit}.", "status": "complete", "done": True},
                        "type": "status"
                    })
                    return formatted_posts
            except Exception as e:
                await __event_emitter__({
                    "data": {"description": f"Failed to retrieve any posts from r/{subreddit}: {e}.", "status": "complete", "done": True},
                    "type": "status"
                })
                return f"Error: {e}"

    async def get_user_feed(
        self,
        username: str,
        __event_emitter__: Callable[[dict], Awaitable[None]],
    ) -> str:
        """
        Retrieves the latest posts from a user's feed.
        """
        cache_path = os.path.join(os.path.dirname(__file__), "reddit_cache.db")
        with shelve.open(cache_path) as cache:
            cache_key = f"user_{username}"
            cached_item = cache.get(cache_key)

            if cached_item and not is_cache_stale(cached_item["timestamp"]):
                await __event_emitter__({
                    "data": {"description": f"Returning cached posts for u/{username}...", "status": "complete", "done": True},
                    "type": "status"
                })
                return cached_item["data"]

            try:
                await __event_emitter__({
                    "data": {"description": f"Retrieving latest posts from u/{username}...", "status": "running"},
                    "type": "status"
                })
                response = requests.get(
                    f"https://www.reddit.com/user/{username}.json",
                    headers={
                        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
                    },
                )
                if response.status_code != 200:
                    await __event_emitter__({
                        "data": {"description": f"Failed to retrieve any posts from u/{username}'s Reddit Feed: {response.status_code}.", "status": "complete", "done": True},
                        "type": "status"
                    })
                    return f"Error: {response.status_code}"
                else:
                    page = parse_reddit_page(response)
                    posts = parse_posts(page)
                    formatted_posts = format_posts_to_human_readable(posts)
                    update_cache(cache, cache_key, formatted_posts)

                    await __event_emitter__({
                        "data": {"description": f"Retrieved {len(posts)} posts from u/{username}'s Reddit Feed.", "status": "complete", "done": True},
                        "type": "status"
                    })
                    return formatted_posts
            except Exception as e:
                await __event_emitter__({
                    "data": {"description": f"Failed to retrieve any posts from u/{username}'s Reddit Feed: {e}.", "status": "complete", "done": True},
                    "type": "status"
                })
                return f"Error: {e}"



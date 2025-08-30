"""
title: Reddit Tool
description: A tool to fetch posts from subreddits, users, and to perform searches across Reddit. Includes caching and content filtering.
author: @nathanwindisch (Refactored by Gemini & Sergii Zabigailo)
version: 1.1.0
"""

import re
import json
import requests
import os
import shelve
from typing import Awaitable, Callable, List
from pydantic import BaseModel, Field
from requests.models import Response

# --- Caching Setup ---

def is_cache_stale(cache_entry: dict, ttl_seconds: int = 3600) -> bool:
    """Check if a cache entry is older than the TTL."""
    import time
    if "timestamp" not in cache_entry:
        return True
    return (time.time() - cache_entry["timestamp"]) > ttl_seconds

# --- Data Parsing Functions ---

def parse_reddit_page(response: Response) -> list:
    """Safely parse the main JSON structure of a Reddit API response."""
    try:
        data = response.json()
        return data.get("data", {}).get("children", [])
    except json.JSONDecodeError:
        return []

def parse_posts(data: list) -> list:
    """Parse a list of post items into a structured format, with robust filtering."""
    posts = []
    for item in data:
        if item.get("kind") != "t3":
            continue
        item_data = item.get("data", {})
        domain = item_data.get("domain", "").lower()
        url = item_data.get("url", "").lower()

        # Exclude posts from known image, video, and gallery domains/URLs
        if domain in ['i.redd.it', 'youtube.com', 'youtu.be', 'v.redd.it']:
            continue
        if 'reddit.com/gallery/' in url:
            continue

        # As a fallback, also check the URL extension for images from other domains
        if url.endswith(('.png', '.jpg', '.jpeg', '.gif', '.webp')):
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
                "upvotes": item_data.get("ups", 0),
                "downvotes": item_data.get("downs", 0),
                "total_comments": item_data.get("num_comments", 0),
                "is_self": item_data.get("is_self", False),
                "published_at": item_data.get("created_utc"),
            }
        )
    return posts

def parse_comments(data: list) -> list:
    """Parse a list of comment items into a structured format."""
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
            }
        )
    return comments

# --- Formatting and Summarization Functions ---

def summarize_text(text: str, max_length: int = 400) -> str:
    """Create an intelligent summary of text by preserving the beginning and end."""
    if not text or len(text) <= max_length:
        return text

    sentences = re.split(r'(?<=[.!?])\s+', text)
    if not sentences:
        return text[:max_length]

    summary = sentences[0]
    # Add sentences from the middle until we approach the max_length
    for sentence in sentences[1:-1]:
        if len(summary) + len(sentence) + 5 < max_length: # 5 for " [...] "
            summary += " " + sentence
        else:
            break
    
    # Always try to include the last sentence
    if len(sentences) > 1 and len(summary) + len(sentences[-1]) + 5 < max_length:
         summary += " [...] " + sentences[-1]
    else:
        summary += "..."

    return summary.strip()


def format_posts_to_human_readable(posts: list) -> str:
    """Format a list of posts into a clean, readable string."""
    if not posts:
        return "No relevant posts found."
    output = []
    for post in posts:
        raw_description = (post.get('description', '') or '').strip()
        # Remove URLs to save token space and provide cleaner context for the LLM
        description_no_urls = re.sub(r'https?://\S+', '', raw_description).strip()
        
        description_part = ""
        if description_no_urls:
            summary = summarize_text(description_no_urls, 400)
            description_part = f"\nDescription: {summary}"

        output.append(
            f"Title: {post.get('title', 'N/A')}\n"
            f"Subreddit: r/{post.get('subreddit_name', 'N/A')}\n"
            f"Author: u/{post.get('author_username', 'N/A')}\n"
            f"Score: {post.get('score', 0)} | Comments: {post.get('total_comments', 0)}\n"
            f"Link: {post.get('link', '#')}"
            f"{description_part}"
        )
    return "\n---\n".join(output)

def format_comments_to_human_readable(comments: list) -> str:
    """Format a list of comments into a clean, readable string."""
    if not comments:
        return "No comments found."
    output = []
    for comment in comments:
        raw_body = (comment.get("body", "") or "").strip()
        # Remove URLs to save token space and provide cleaner context for the LLM
        body_no_urls = re.sub(r'https?://\S+', '', raw_body).strip()
        summary = summarize_text(body_no_urls, 250) # Shorter summary for comments
        output.append(
            f"Author: u/{comment.get('author_username', 'N/A')} | Score: {comment.get('score', 0)}\n"
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
                        "properties": {"subreddit": {"type": "string", "description": "The name of the subreddit (e.g., 'stocks')."}},
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
                        "properties": {"username": {"type": "string", "description": "The Reddit username (e.g., 'gregoleg')."}},
                        "required": ["username"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "search_reddit",
                    "description": "Search for posts containing specific keywords, either across all of Reddit or within a specific subreddit.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "The search term or keywords (e.g., 'AAPL')."},
                            "subreddit": {"type": "string", "description": "Optional. The subreddit to search within. Use 'all' or 'r' for a site-wide search.", "default": "all"},
                        },
                        "required": ["query"],
                    },
                },
            },
        ]
        self.cache_path = os.path.join(os.path.dirname(__file__), "reddit_cache.db")

    async def _make_request(self, url: str, headers: dict, cache_key: str, __event_emitter__):
        with shelve.open(self.cache_path) as cache:
            if cache_key in cache and not is_cache_stale(cache[cache_key]):
                if __event_emitter__:
                    await __event_emitter__({"type": "status", "data": {"description": f"Loading '{cache_key}' from cache.", "done": False}})
                return cache[cache_key]["data"]

            try:
                response = requests.get(url, headers=headers)
                if not response.ok:
                    return f"Error: {response.status_code}"
                
                parsed_data = parse_reddit_page(response)
                cache[cache_key] = {"data": parsed_data, "timestamp": __import__("time").time()}
                return parsed_data
            except Exception as e:
                return f"Error: {e}"

    async def get_subreddit_feed(self, subreddit: str, __user__={}, __event_emitter__=None) -> str:
        if not subreddit:
            return "Error: Subreddit name not provided."
        subreddit = subreddit.replace("/r/", "").replace("r/", "")
        url = f"https://reddit.com/r/{subreddit}.json"
        cache_key = f"subreddit_{subreddit}"
        headers = {"User-Agent": __user__.get("valves", self.UserValves()).USER_AGENT}
        
        if __event_emitter__:
            await __event_emitter__({"type": "status", "data": {"description": f"Fetching feed for r/{subreddit}...", "done": False}})
        result = await self._make_request(url, headers, cache_key, __event_emitter__)

        if isinstance(result, str): # It's an error message
            if __event_emitter__:
                await __event_emitter__({"type": "status", "data": {"description": f"Failed to get feed for r/{subreddit}.", "done": True}})
            return result
        
        posts = parse_posts(result)
        if __event_emitter__:
            await __event_emitter__({"type": "status", "data": {"description": f"Feed for r/{subreddit} retrieved.", "done": True}})
        return format_posts_to_human_readable(posts)

    async def get_user_feed(self, username: str, __user__={}, __event_emitter__=None) -> str:
        if not username:
            return "Error: Username not provided."
        username = username.replace("/u/", "").replace("u/", "")
        url = f"https://reddit.com/u/{username}.json"
        cache_key = f"user_{username}"
        headers = {"User-Agent": __user__.get("valves", self.UserValves()).USER_AGENT}

        if __event_emitter__:
            await __event_emitter__({"type": "status", "data": {"description": f"Fetching feed for u/{username}...", "done": False}})
        result = await self._make_request(url, headers, cache_key, __event_emitter__)

        if isinstance(result, str):
            if __event_emitter__:
                await __event_emitter__({"type": "status", "data": {"description": f"Failed to get feed for u/{username}.", "done": True}})
            return result
            
        posts = parse_posts(result)
        comments = parse_comments(result)
        
        formatted_posts = f"--- POSTS ---\n{format_posts_to_human_readable(posts)}" if posts else ""
        formatted_comments = f"\n\n--- COMMENTS ---\n{format_comments_to_human_readable(comments)}" if comments else ""

        if __event_emitter__:
            await __event_emitter__({"type": "status", "data": {"description": f"Feed for u/{username} retrieved.", "done": True}})
        if not formatted_posts and not formatted_comments:
            return "No activity found for this user."
            
        return f"{formatted_posts}{formatted_comments}".strip()

    async def search_reddit(self, query: str, subreddit: str = "all", __user__={}, __event_emitter__=None) -> str:
        headers = {"User-Agent": __user__.get("valves", self.UserValves()).USER_AGENT}
        
        if subreddit.lower() in ["r", "all", ""]:
            url = f"https://www.reddit.com/search.json?q={query}"
            search_scope = "all of Reddit"
        else:
            subreddit = subreddit.replace("/r/", "").replace("r/", "")
            url = f"https://www.reddit.com/r/{subreddit}/search.json?q={query}&restrict_sr=on"
            search_scope = f"r/{subreddit}"
            
        cache_key = f"v2_search_{subreddit}_{query}"

        if __event_emitter__:
            await __event_emitter__({"type": "status", "data": {"description": f"Searching for '{query}' in {search_scope}...", "done": False}})
        result = await self._make_request(url, headers, cache_key, __event_emitter__)

        if isinstance(result, str):
            if __event_emitter__:
                await __event_emitter__({"type": "status", "data": {"description": f"Search for '{query}' failed.", "done": True}})
            return result

        posts = parse_posts(result)
        if __event_emitter__:
            await __event_emitter__({"type": "status", "data": {"description": f"Search for '{query}' complete.", "done": True}})
        return format_posts_to_human_readable(posts)

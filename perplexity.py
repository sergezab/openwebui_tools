"""
title: Perplexity Web Search Tool
author: Sergii Zabigailo
version: 0.2.0
license: MIT
description: A revised version of the Perplexity tool with a more effective function description to ensure it gets called by the LLM.
"""

from pydantic import BaseModel, Field
from typing import Optional, Callable, Any, Dict, List, Union, Tuple
import requests
import asyncio
import os
import json
import logging
from datetime import datetime, timedelta
from sentence_transformers import SentenceTransformer
import numpy as np
import httpx # Use httpx for async requests

# --- Logging Configuration (No changes needed here, it's well-configured) ---
logger = logging.getLogger(__name__)
if not logger.handlers:
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    log_dir = os.path.expanduser("~")
    log_file = os.path.join(log_dir, "perplexity.log")
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    file_handler = logging.FileHandler(log_file, mode="a")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)


# --- Caching and Similarity Logic (No changes needed here, it's excellent) ---
_model = None

def _get_model():
    """Lazy load the sentence transformer model"""
    global _model
    if _model is None:
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model

def _calculate_similarity(query1: str, query2: str) -> float:
    """Calculate cosine similarity between two queries"""
    model = _get_model()
    embedding1 = model.encode([query1])[0]
    embedding2 = model.encode([query2])[0]
    return float(
        np.dot(embedding1, embedding2)
        / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))
    )

def _find_similar_query(
    query: str, cache: Dict[str, Any], similarity_threshold: float = 0.85
) -> Tuple[Optional[str], Optional[Dict]]:
    """Find a similar query in the cache above the similarity threshold"""
    for cached_query, cached_data in cache.items():
        similarity = _calculate_similarity(query, cached_query)
        if similarity >= similarity_threshold:
            logger.info(f"Found similar query in cache. Similarity: {similarity:.2f}")
            return cached_query, cached_data
    return None, None

def _load_cache(cache_file: str) -> Dict[str, Any]:
    """Load cached search results from file and remove outdated entries"""
    if os.path.exists(cache_file):
        try:
            with open(cache_file, "r") as f:
                cache = json.load(f)
            valid_cache = {
                query: data for query, data in cache.items() if _is_cache_valid(data)
            }
            if len(valid_cache) < len(cache):
                logger.info(f"Removed {len(cache) - len(valid_cache)} outdated entries from cache")
                _save_cache(cache_file, valid_cache)
            return valid_cache
        except json.JSONDecodeError as e:
            logger.error(f"Error loading cache from {cache_file}: {str(e)}")
            return {}
    return {}

def _save_cache(cache_file: str, cache_data: Dict[str, Any]) -> None:
    """Save cached search results to file"""
    try:
        os.makedirs(os.path.dirname(cache_file), exist_ok=True)
        with open(cache_file, "w") as f:
            json.dump(cache_data, f, indent=4)
    except Exception as e:
        logger.error(f"Error saving cache to {cache_file}: {str(e)}")

def _is_cache_valid(
    cached_data: Dict[str, Any], cache_duration: timedelta = timedelta(hours=1)
) -> bool:
    """Check if cached data is still valid based on timestamp"""
    if "timestamp" not in cached_data:
        return False
    try:
        cache_time = datetime.fromisoformat(cached_data["timestamp"])
        return datetime.now() - cache_time <= cache_duration
    except Exception as e:
        logger.error(f"Error validating cache timestamp: {str(e)}")
        return False


class Tools:
    class Valves(BaseModel):
        PERPLEXITY_API_KEY: str = Field(
            default="", description="Required API key to access Perplexity services"
        )
        PERPLEXITY_API_BASE_URL: str = Field(
            default="https://api.perplexity.ai",
            description="The base URL for Perplexity API endpoints",
        )

    def __init__(self):
        self.valves = self.Valves()
        self.citation = True
        
        # *** REVISED TOOL DEFINITION ***
        self.tools = [
            {
                "type": "function",
                "function": {
                    "name": "web_search",
                    "description": (
                        "Use this tool to get real-time, up-to-the-minute information from the web. "
                        "It is essential for topics like today's news, current events, weather, "
                        "recent sports scores, or the latest stock prices."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "The precise question or topic to search for. For example: 'latest news on AI developments' or 'who won the F1 race yesterday?'"
                            }
                        },
                        "required": ["query"],
                    },
                },
            }
        ]
    
    # This is the function that will be called by the LLM
    async def web_search(
        self, query: str, __event_emitter__: Optional[Callable[[Dict], Any]] = None
    ) -> str:
        if not self.valves.PERPLEXITY_API_KEY:
            return "Error: PERPLEXITY_API_KEY not provided."

        async def emit_status(description: str, status: str = "in_progress", done: bool = False):
            if __event_emitter__:
                await __event_emitter__({
                    "type": "status",
                    "data": {"description": description, "status": status, "done": done},
                })

        try:
            cache_file = os.path.join(os.path.dirname(__file__), "perplexity_cache.json")
            cache = _load_cache(cache_file)

            similar_query, similar_cache = _find_similar_query(query, cache)
            if similar_query and _is_cache_valid(cache[similar_query]):
                logger.info(f"Using cached result for similar query: {similar_query}")
                await emit_status(f"Found similar item in cache: '{similar_query}'", "cached", True)
                return cache[similar_query]["data"]["formatted_response"]
            
            await emit_status(f"Searching the web for: {query}", "in_progress")

            headers = {
                "Authorization": f"Bearer {self.valves.PERPLEXITY_API_KEY}",
                "Content-Type": "application/json",
                "accept": "application/json",
            }
            payload = {
                "model": "sonar", # Corrected model name based on Perplexity's example.
                "messages": [
                    {"role": "user", "content": query},
                ],
            }
            
            # Using httpx for a non-blocking request in an async function
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.valves.PERPLEXITY_API_BASE_URL}/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=30,
                )
                response.raise_for_status()
                result = response.json()

            content = result["choices"][0]["message"]["content"]
            response_text = f"**Search Results for:** '{query}'\n\n---\n\n{content}"

            cache[query] = {
                "data": {
                    "content": content,
                    "formatted_response": response_text,
                },
                "timestamp": datetime.now().isoformat(),
            }
            _save_cache(cache_file, cache)
            logger.info(f"Cached new search result for query: {query}")
            
            await emit_status("Search complete.", "complete", True)
            return response_text

        except httpx.HTTPStatusError as e:
            error_details = f"HTTP Error: {e.response.status_code} - Response: {e.response.text}"
            logger.error(f"Search error: {error_details}")
            await emit_status(error_details, "error", True)
            return f"Error performing web search: {error_details}"
        except Exception as e:
            error_msg = f"An unexpected error occurred: {str(e)}"
            logger.error(f"Search error: {error_msg}")
            await emit_status(error_msg, "error", True)
            return error_msg




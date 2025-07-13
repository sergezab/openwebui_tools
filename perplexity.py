"""
title: Perplexity Web Search Tool
author: Sergii Zabigailo
version: 0.1.0
license: MIT
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

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Create formatter
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

# Create a FileHandler
log_dir = os.path.expanduser("~")
log_file = os.path.join(log_dir, "perplexity.log")
os.makedirs(os.path.dirname(log_file), exist_ok=True)
file_handler = logging.FileHandler(log_file, mode="a")
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(formatter)

# Create a StreamHandler (console)
stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.INFO)
stream_handler.setFormatter(formatter)

# Clear any existing handlers
if logger.hasHandlers():
    logger.handlers.clear()

# Add handlers
logger.addHandler(file_handler)
logger.addHandler(stream_handler)

# Initialize the sentence transformer model
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

            # Remove outdated entries
            valid_cache = {
                query: data for query, data in cache.items() if _is_cache_valid(data)
            }

            # If entries were removed, save the cleaned cache
            if len(valid_cache) < len(cache):
                logger.info(
                    f"Removed {len(cache) - len(valid_cache)} outdated entries from cache"
                )
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
            json.dump(cache_data, f)
    except Exception as e:
        logger.error(f"Error saving cache to {cache_file}: {str(e)}")
        raise


def _is_cache_valid(
    cached_data: Dict[str, Any], cache_duration: timedelta = timedelta(hours=1)
) -> bool:
    """Check if cached data is still valid based on timestamp"""
    if not cached_data or "timestamp" not in cached_data:
        return False
    try:
        cache_time = datetime.fromisoformat(cached_data["timestamp"])
        return datetime.now() - cache_time <= cache_duration
    except Exception as e:
        logger.error(f"Error validating cache: {str(e)}")
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
        self.tools = [
            {
                "type": "function",
                "function": {
                    "name": "web_search",
                    "description": "Search the web using Perplexity AI",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "The search query to look up",
                            }
                        },
                        "required": ["query"],
                    },
                },
            }
        ]

    async def web_search(
        self, query: str, __event_emitter__: Optional[Callable[[Dict], Any]] = None
    ) -> str:
        if not self.valves.PERPLEXITY_API_KEY:
            raise Exception("PERPLEXITY_API_KEY not provided in valves")

        # Status emitter helper
        async def emit_status(
            description: str, status: str = "in_progress", done: bool = False
        ):
            if __event_emitter__:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {
                            "description": description,
                            "status": status,
                            "done": done,
                        },
                    }
                )

        try:
            # Initialize cache
            cache_file = os.path.join(
                os.path.dirname(__file__), "perplexity_cache.json"
            )
            cache = _load_cache(cache_file)

            # Check cache for this query or similar queries
            similar_query, similar_cache = _find_similar_query(query, cache)
            if similar_query and _is_cache_valid(cache[similar_query]):
                logger.info(f"Using cached result for similar query: {similar_query}")
                await emit_status(
                    "Retrieved from cache (similar query match)", "cached", False
                )
                cached_result = cache[similar_query]["data"]

                # Emit citations from cache
                if __event_emitter__:
                    await __event_emitter__(
                        {
                            "type": "citation",
                            "data": {
                                "document": [cached_result["content"]],
                                "metadata": [
                                    {"source": "Perplexity AI Search (Cached)"}
                                ],
                                "source": {"name": "Perplexity AI"},
                            },
                        }
                    )
                    for url in cached_result.get("citations", []):
                        await __event_emitter__(
                            {
                                "type": "citation",
                                "data": {
                                    "document": [cached_result["content"]],
                                    "metadata": [{"source": url}],
                                    "source": {"name": url},
                                },
                            }
                        )

                await emit_status(
                    "Retrieved cached results", status="complete", done=True
                )
                return cached_result["formatted_response"]

            # If not in cache or cache invalid, perform new search
            await emit_status(f"Asking Perplexity: {query}", "searching")

            headers = {
                "Authorization": f"Bearer {self.valves.PERPLEXITY_API_KEY}",
                "Content-Type": "application/json",
            }

            payload = {
                "model": "llama-3.1-sonar-small-128k-online",
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a helpful search assistant. Provide concise and accurate information.",
                    },
                    {"role": "user", "content": query},
                ],
            }

            await emit_status("Processing search results...", "processing")

            response = requests.post(
                f"{self.valves.PERPLEXITY_API_BASE_URL}/chat/completions",
                headers=headers,
                json=payload,
                timeout=30,
            )
            response.raise_for_status()
            result = response.json()

            content = result["choices"][0]["message"]["content"]
            citations = result.get("citations", [])

            # Emit Perplexity as primary source
            if __event_emitter__:
                await __event_emitter__(
                    {
                        "type": "citation",
                        "data": {
                            "document": [content],
                            "metadata": [{"source": "Perplexity AI Search"}],
                            "source": {"name": "Perplexity AI"},
                        },
                    }
                )

            # Emit each URL citation
            if citations and __event_emitter__:
                for url in citations:
                    await __event_emitter__(
                        {
                            "type": "citation",
                            "data": {
                                "document": [content],
                                "metadata": [{"source": url}],
                                "source": {"name": url},
                            },
                        }
                    )

            # Format response with citations
            original_prompt = f"Original query: {query}\n\n"
            response_text = original_prompt + f"{content}\n\nSources:\n"
            response_text += "- Perplexity AI Search\n"

            for url in citations:
                response_text += f"- {url}\n"

            # Cache the result
            cache[query] = {
                "data": {
                    "content": content,
                    "citations": citations,
                    "formatted_response": response_text,
                },
                "timestamp": datetime.now().isoformat(),
            }
            _save_cache(cache_file, cache)
            logger.info(f"Cached search result for query: {query}")

            # Complete status
            await emit_status(
                "Search completed successfully", status="complete", done=True
            )

            return response_text

        except Exception as e:
            error_msg = f"Error performing web search: {str(e)}"
            logger.error(f"Search error: {error_msg}")
            await emit_status(error_msg, status="error", done=True)
            return error_msg

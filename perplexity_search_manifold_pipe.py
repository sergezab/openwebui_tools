"""
title: Perplexity Manifold Pipes
authors: dotJustin
author_url: https://github.com/dot-Justin
funding_url: https://github.com/open-webui
version: 0.1.0
license: MIT
"""

"""
This uses the reverse engineered Perplexity.ai API, so there is no API key required. Just enable and use!
If you have issues, ping me inside the open-webui discord server: @dotjustin
"""

import requests
import json
import re
import time
from typing import Union, Generator, Iterator
from pydantic import BaseModel, Field
from concurrent.futures import ThreadPoolExecutor, as_completed


class Pipe:
    def __init__(self):
        self.type = "manifold"
        self.id = "perplexity"
        self.name = "Perplexity/"
        self.base_url = "https://www.perplexity.ai"
        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
                "Accept": "application/json",
                "Origin": "https://www.perplexity.ai",
                "Referer": "https://www.perplexity.ai/",
            }
        )

    def get_search_focuses(self):
        return ["internet", "scholar", "writing", "math", "coding"]

    def pipes(self):
        return [
            {"id": f"perplexity.{focus}", "name": f"{focus.capitalize()}"}
            for focus in self.get_search_focuses()
        ]

    def get_socket_id(self):
        url = f"{self.base_url}/socket.io/?EIO=4&transport=polling"
        try:
            response = self.session.get(url)
            sid_match = re.search(r'"sid":"([^"]+)"', response.text)
            if not sid_match:
                time.sleep(1)
                response = self.session.get(url)
                sid_match = re.search(r'"sid":"([^"]+)"', response.text)
            return sid_match.group(1) if sid_match else None
        except Exception as e:
            print(f"Socket ID error: {e}")
            return None

    def authenticate(self, sid):
        auth_url = f"{self.base_url}/socket.io/?EIO=3&transport=polling&sid={sid}"
        try:
            response = self.session.post(
                auth_url, data='40{"jwt":"anonymous-ask-user"}'
            )
            if response.status_code != 200:
                time.sleep(1)
                response = self.session.post(
                    auth_url, data='40{"jwt":"anonymous-ask-user"}'
                )
            return response.status_code == 200
        except Exception as e:
            print(f"Auth error: {e}")
            return False

    def _process_message(self, message):
        while message and message[0].isdigit():
            message = message[1:]

        try:
            data = json.loads(message)
            if isinstance(data, list):
                if data[0] == "query_progress":
                    data = data[1:]

                if data and isinstance(data[0], dict):
                    return {
                        "status": data[0].get("status"),
                        "text": json.loads(data[0].get("text", "{}")),
                    }
        except (json.JSONDecodeError, IndexError) as e:
            print(f"Process message error: {e}")
            return None

    def _fetch_title(self, url):
        try:
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                match = re.search(
                    r"<title>(.*?)</title>", response.text, re.IGNORECASE | re.DOTALL
                )
                if match:
                    title = match.group(1).strip()
                    return title
            return "No Title"
        except Exception as e:
            print(f"Error fetching title for {url}: {e}")
            return "No Title"

    def _format_answer_with_sources(self, answer: str, web_results: list) -> str:

        source_urls = {}
        for i, source in enumerate(web_results, 1):
            source_urls[f"[{i}]"] = source.get("url", "")

        for indicator, url in source_urls.items():
            if url:  # Only create hyperlink if URL exists
                answer = answer.replace(indicator, f"[{indicator}]({url})")

        with ThreadPoolExecutor(max_workers=5) as executor:
            future_to_url = {
                executor.submit(self._fetch_title, source.get("url", "")): i
                for i, source in enumerate(web_results)
                if not source.get("title")
            }
            for future in as_completed(future_to_url):
                i = future_to_url[future]
                try:
                    title = future.result()
                    web_results[i]["title"] = title
                except Exception as e:
                    print(f"Error in fetching title: {e}")

        # Add sources section at the bottom
        formatted_answer = answer + "\n\nSources:\n"
        for i, source in enumerate(web_results, 1):
            title = source.get("title", "No Title")
            url = source.get("url", "No URL")
            formatted_answer += f"{i}. [{title}]({url})\n"
        return formatted_answer

    async def perform_search(
        self, query: str, search_focus: str, __event_emitter__=None
    ) -> str:
        retries = 3

        for attempt in range(retries):
            try:
                if __event_emitter__:
                    await __event_emitter__(
                        {
                            "type": "status",
                            "data": {
                                "description": "Connecting to Perplexity...",
                                "done": False,
                            },
                        }
                    )

                sid = self.get_socket_id()
                if not sid:
                    if attempt == retries - 1:
                        return "Error: Failed to get socket ID"
                    time.sleep(1)
                    continue

                if __event_emitter__:
                    await __event_emitter__(
                        {
                            "type": "status",
                            "data": {
                                "description": "Authenticating search request...",
                                "done": False,
                            },
                        }
                    )

                if not self.authenticate(sid):
                    if attempt == retries - 1:
                        return "Error: Authentication failed"
                    time.sleep(1)
                    continue

                poll_url = (
                    f"{self.base_url}/socket.io/?EIO=4&transport=polling&sid={sid}"
                )
                search_payload = json.dumps(
                    [
                        "perplexity_ask",
                        query,
                        {
                            "version": "2.5",
                            "search_focus": search_focus,
                            "mode": "concise",
                            "prompt_source": "user",
                            "query_source": "home",
                        },
                    ]
                )

                if __event_emitter__:
                    await __event_emitter__(
                        {
                            "type": "status",
                            "data": {
                                "description": "Waiting for response from Perplexity...",
                                "done": False,
                            },
                        }
                    )

                post_response = self.session.post(poll_url, data=f"421{search_payload}")
                if post_response.status_code != 200:
                    time.sleep(1)
                    continue

                if __event_emitter__:
                    await __event_emitter__(
                        {
                            "type": "status",
                            "data": {
                                "description": "Processing search results...",
                                "done": False,
                            },
                        }
                    )

                time.sleep(1)
                max_attempts = 20
                for _ in range(max_attempts):
                    response = self.session.get(poll_url)
                    if not response.text:
                        time.sleep(0.5)
                        continue

                    messages = response.text.split("\x1e")
                    for message in messages:
                        result = self._process_message(message)
                        if result and result.get("status") == "completed":
                            answer = result["text"].get("answer", "")
                            web_results = result["text"].get("web_results", [])
                            if not answer and not web_results:
                                break

                            if __event_emitter__:
                                await __event_emitter__(
                                    {
                                        "type": "status",
                                        "data": {
                                            "description": "Search complete",
                                            "done": True,
                                        },
                                    }
                                )

                            return self._format_answer_with_sources(answer, web_results)
                    time.sleep(0.5)

            except Exception as e:
                print(f"Search error (attempt {attempt + 1}): {e}")
                if attempt == retries - 1:
                    return f"Error performing search: {str(e)}"
                time.sleep(1)
                continue

        return "Error: Search timed out"

    async def pipe(
        self, body: dict, __event_emitter__=None
    ) -> Union[str, Generator, Iterator]:
        try:
            query = body.get("messages", [{}])[-1].get("content", "")
            if not query:
                return "Please provide a search query."
            if len(query.strip()) < 3:
                return "Please provide a longer search query."

            search_focus = body["model"].split(".")[-1]

            result = await self.perform_search(query, search_focus, __event_emitter__)

            if result.startswith("Error"):
                print(f"Search failed: {result}")
            return result

        except Exception as e:
            print(f"Pipe error: {e}")
            return f"Error in pipe execution: {str(e)}"

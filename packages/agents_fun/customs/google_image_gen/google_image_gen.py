# -*- coding: utf-8 -*-
# ------------------------------------------------------------------------------
#
#   Copyright 2024 Valory AG
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
#
# ------------------------------------------------------------------------------
"""This module contains the implementation of the google_image_gen tool based on a working snippet."""

import functools
import json
import os
from io import BytesIO
from typing import Any, Callable, Dict, Optional, Tuple

import anthropic
from google import genai  # Use import style from working snippet
import openai
from aea_cli_ipfs.ipfs_utils import IPFSTool
from google.api_core import exceptions as google_exceptions
from google.genai import types  # Keep this for types.GenerateContentConfig
from PIL import Image

# Define MechResponse type alias matching the other tools
MechResponse = Tuple[str, Optional[str], Optional[Dict[str, Any]], Any, Any]

# Define allowed tools for this module
ALLOWED_TOOLS = [
    "google-image-gen",
]


# Replicate the key rotation decorator from other tools
def with_key_rotation(func: Callable):
    """Decorator for handling API key rotation and retries."""

    @functools.wraps(func)
    def wrapper(*args, **kwargs) -> MechResponse:
        api_keys = kwargs["api_keys"]
        # Ensure api_keys object has the expected methods
        if (
            not hasattr(api_keys, "max_retries")
            or not hasattr(api_keys, "rotate")
            or not hasattr(api_keys, "get")
        ):
            error_msg = "api_keys object does not have required methods (max_retries, rotate, get)"
            prompt_val = kwargs.get("prompt", "N/A")
            callback_val = kwargs.get("counter_callback", None)
            return error_msg, prompt_val, None, callback_val, None  # Return 5 elements

        retries_left: Dict[str, int] = api_keys.max_retries()

        def execute() -> MechResponse:
            """Execute the function with retries."""
            try:
                result = func(*args, **kwargs)
                return result + (api_keys,)
            except (
                anthropic.RateLimitError,
                openai.RateLimitError,
                google_exceptions.ResourceExhausted,
                google_exceptions.TooManyRequests,
            ) as e:
                service = "google_api_key"
                if isinstance(e, anthropic.RateLimitError):
                    service = "anthropic"
                elif isinstance(e, openai.RateLimitError):
                    if retries_left["openai"] <= 0 and retries_left["openrouter"] <= 0:
                        raise e
                    retries_left["openai"] -= 1
                    retries_left["openrouter"] -= 1
                    api_keys.rotate("openai")
                    api_keys.rotate("openrouter")
                    return execute()

                if retries_left.get(service, 0) <= 0:
                    print(f"No retries left for service: {service}")
                    raise e

                retries_left[service] -= 1
                print(
                    f"Rate limit error for {service}. Retries left: {retries_left[service]}. Rotating key."
                )
                api_keys.rotate(service)
                return execute()
            except Exception as e:
                print(f"An unexpected error occurred: {e}")
                error_response = str(e)
                prompt_value = kwargs.get(
                    "prompt", "Prompt not available in error context"
                )
                callback_value = kwargs.get("counter_callback", None)
                return error_response, prompt_value, None, callback_value, api_keys

        return execute()

    return wrapper


@with_key_rotation
def run(**kwargs) -> Tuple[str, Optional[str], Optional[Dict[str, Any]], Any]:
    """Runs the Google image generation task using genai.Client."""
    prompt = kwargs["prompt"]
    # Use the key name expected by the test script and .env
    api_key = kwargs["api_keys"].get(
        "google_api_key"
    )  # Corresponds to GEMINI_API_KEY in test
    tool = kwargs.get("tool")
    counter_callback = kwargs.get("counter_callback", None)
    # Use model from working snippet
    model_name = "gemini-2.0-flash-exp-image-generation"

    if tool not in ALLOWED_TOOLS:
        return (
            f"Tool {tool} is not supported by this agent.",
            prompt,
            None,
            counter_callback,
        )

    if not api_key:
        return (
            "Google API key (GEMINI_API_KEY) not provided.",
            prompt,
            None,
            counter_callback,
        )

    try:
        # Initialize client using the working snippet method
        client = genai.Client(api_key=api_key)

        # Generate content using client.models.generate_content
        response = client.models.generate_content(
            model=model_name,
            contents=prompt,
            # Request both Text and Image modalities as per working snippet
            config=types.GenerateContentConfig(response_modalities=["Text", "Image"]),
            # generation_config is part of GenerateContentConfig now if needed
            # generation_config=types.GenerationConfig(candidate_count=1) # Example
        )

        if not response.candidates or not response.candidates[0].content.parts:
            return (
                "No image data found in the response candidates.",
                prompt,
                None,
                counter_callback,
            )

        image_part = None
        for part in response.candidates[0].content.parts:
            # Directly check for inline_data as per working snippet
            if hasattr(part, "inline_data") and part.inline_data.mime_type.startswith(
                "image/"
            ):
                image_part = part.inline_data
                break

        if image_part is None:
            return (
                "No image data found in response parts (checked inline_data).",
                prompt,
                None,
                counter_callback,
            )

        # Process the image data
        image_data = image_part.data
        image = Image.open(BytesIO(image_data))
        # Use a unique temp file name if running tests concurrently
        temp_image_path = f"temp_generated_image_{os.getpid()}.png"
        image.save(temp_image_path)

        # Upload to IPFS
        try:
            ipfs_tool = IPFSTool()
            _, image_hash, _ = ipfs_tool.add(temp_image_path)
        except FileNotFoundError:
            # Clean up temp file even if IPFS fails
            if os.path.exists(temp_image_path):
                os.remove(temp_image_path)
            return (
                "IPFS tool not found or not configured correctly.",
                prompt,
                None,
                counter_callback,
            )
        finally:
            # Ensure temporary file is always removed
            if os.path.exists(temp_image_path):
                os.remove(temp_image_path)

        # Prepare result
        result_data = {"image_hash": image_hash, "prompt": prompt, "model": model_name}
        return json.dumps(result_data), prompt, None, counter_callback

    except google_exceptions.GoogleAPIError as e:
        print(f"Google API error: {e}")
        return f"Google API error: {e}", prompt, None, counter_callback
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        # Ensure temp file is cleaned up on unexpected error if it exists
        if "temp_image_path" in locals() and os.path.exists(temp_image_path):
            try:
                os.remove(temp_image_path)
            except OSError as remove_err:
                print(
                    f"Error removing temp file during exception handling: {remove_err}"
                )
        return f"An error occurred: {e}", prompt, None, counter_callback

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
from google import genai
from google.genai import types
import openai
from aea_cli_ipfs.ipfs_utils import IPFSTool
from google.api_core import exceptions as google_exceptions
from PIL import Image
from googleapiclient.errors import HttpError as GoogleApiClientHttpError


# Define MechResponse type alias matching the other tools
MechResponse = Tuple[str, Optional[str], Optional[Dict[str, Any]], Any, Any]

# Define allowed tools for this module
ALLOWED_TOOLS = [
    "google-imagen",
]


def with_key_rotation(func: Callable):
    @functools.wraps(func)
    def wrapper(*args, **kwargs) -> MechResponse:
        # this is expected to be a KeyChain object,
        # although it is not explicitly typed as such
        api_keys = kwargs["api_keys"]
        retries_left: Dict[str, int] = api_keys.max_retries()

        def execute() -> MechResponse:
            """Retry the function with a new key."""
            try:
                result = func(*args, **kwargs)
                return result + (api_keys,)
            except anthropic.RateLimitError as e:
                # try with a new key again
                service = "anthropic"
                if retries_left[service] <= 0:
                    raise e
                retries_left[service] -= 1
                api_keys.rotate(service)
                return execute()
            except openai.RateLimitError as e:
                # try with a new key again
                if retries_left["openai"] <= 0 and retries_left["openrouter"] <= 0:
                    raise e
                retries_left["openai"] -= 1
                retries_left["openrouter"] -= 1
                api_keys.rotate("openai")
                api_keys.rotate("openrouter")
                return execute()
            except GoogleApiClientHttpError as e:
                # try with a new key again
                rate_limit_exceeded_code = 429
                if e.status_code != rate_limit_exceeded_code:
                    raise e
                service = "google_api_key"
                if retries_left[service] <= 0:
                    raise e
                retries_left[service] -= 1
                api_keys.rotate(service)
                return execute()
            except Exception as e:
                return str(e), "", None, None, api_keys

        mech_response = execute()
        return mech_response

    return wrapper


def _validate_inputs(
    tool: str, api_key: Optional[str], prompt: str, counter_callback: Any
) -> Optional[Tuple[str, str, None, Any]]:
    """Validate tool and API key."""
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
    return None


def _generate_image_from_google_api(
    client: genai.Client, prompt: str, model_name: str, counter_callback: Any
) -> Tuple[Optional[bytes], Optional[Tuple[str, str, None, Any]]]:
    """Generates image data using the Google API and handles initial response validation."""
    response = client.models.generate_images(
        model=model_name,
        prompt=prompt,
        config=types.GenerateImagesConfig(number_of_images=1),
    )

    if not response.generated_images:
        return None, (
            "No image data found in the response (generated_images is empty).",
            prompt,
            None,
            counter_callback,
        )

    first_generated_image = response.generated_images[0]

    if not hasattr(first_generated_image, "image") or not hasattr(
        first_generated_image.image, "image_bytes"
    ):
        return None, (
            "Image data structure is not as expected.",
            prompt,
            None,
            counter_callback,
        )
    return first_generated_image.image.image_bytes, None


def _save_image_and_upload_to_ipfs(
    image_data: bytes, prompt: str, model_name: str, counter_callback: Any
) -> Tuple[str, Optional[str], Optional[Dict[str, Any]], Any]:
    """Saves the image data to a temporary file, uploads to IPFS, and cleans up."""
    temp_image_path = f"temp_generated_image_{os.getpid()}.png"
    try:
        image = Image.open(BytesIO(image_data))
        image.save(temp_image_path)

        ipfs_tool = IPFSTool()
        _, image_hash, _ = ipfs_tool.add(temp_image_path)

        result_data = {"image_hash": image_hash, "prompt": prompt, "model": model_name}
        return json.dumps(result_data), prompt, None, counter_callback
    except FileNotFoundError:
        return (
            "IPFS tool not found or not configured correctly.",
            prompt,
            None,
            counter_callback,
        )
    finally:
        if os.path.exists(temp_image_path):
            os.remove(temp_image_path)


@with_key_rotation
def run(**kwargs) -> Tuple[str, Optional[str], Optional[Dict[str, Any]], Any]:
    """Runs the Google image generation task using genai.Client."""
    prompt = kwargs["prompt"]
    api_keys = kwargs["api_keys"]
    api_key = api_keys.get("gemini_api_key")
    tool = kwargs.get("tool")
    counter_callback = kwargs.get("counter_callback", None)
    model_name = "imagen-3.0-generate-002"

    validation_error = _validate_inputs(tool, api_key, prompt, counter_callback)
    if validation_error:
        return validation_error

    try:
        client = genai.Client(api_key=api_key)

        image_data, error_response = _generate_image_from_google_api(
            client, prompt, model_name, counter_callback
        )
        if error_response:
            return error_response

        if (
            image_data is None
        ):  # Should not happen if error_response is None, but as a safeguard
            return (
                "Failed to generate image data without specific error.",
                prompt,
                None,
                counter_callback,
            )

        return _save_image_and_upload_to_ipfs(
            image_data, prompt, model_name, counter_callback
        )

    except google_exceptions.GoogleAPIError as e:
        print(f"Google API error: {e}")
        return f"Google API error: {e}", prompt, None, counter_callback
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        # Ensure temp file is cleaned up on unexpected error if it exists
        # temp_image_path is local to _save_image_and_upload_to_ipfs, so direct access here is not possible
        # Consider if this cleanup is still needed here or if it's fully handled in _save_image_and_upload_to_ipfs
        return f"An error occurred: {e}", prompt, None, counter_callback

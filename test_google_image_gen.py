import os
import json
from typing import Dict, Optional
from dotenv import load_dotenv

# Import the run function from your tool
from packages.agents_fun.customs.google_image_gen.google_image_gen import run


# Minimal KeyChain class to mimic the expected structure
class KeyChain:
    def __init__(self, api_keys: Dict[str, Optional[str]]):
        self._keys = api_keys
        self._max_retries = {k: 1 for k in api_keys}  # Simple retry count

    def get(self, key_name: str) -> Optional[str]:
        return self._keys.get(key_name)

    def max_retries(self) -> Dict[str, int]:
        # Return a copy to prevent modification
        return self._max_retries.copy()

    def rotate(self, service: str):
        # Placeholder for rotation logic if needed for testing complex scenarios
        print(f"[KeyChain] Rotating key for {service} (placeholder)")
        # In a real scenario, this would fetch a new key
        pass


# Test the google_image_gen tool
if __name__ == "__main__":

    load_dotenv()

    # Get API key from environment variable
    google_api_key = os.getenv("GEMINI_API_KEY")
    print(f"Google API Key: {google_api_key}")
    if not google_api_key:
        print("Error: GEMINI_API_KEY environment variable not set.")
        exit(1)

    # Create the KeyChain object
    api_keys = KeyChain({"google_api_key": google_api_key})

    # Sample input parameters
    kwargs = {
        "prompt": "Luffy in cyberpunk 2077 universe, please return image in square format",
        "api_keys": api_keys,
        "tool": "google-image-gen",  # Make sure this matches ALLOWED_TOOLS
        "counter_callback": None,  # Optional callback if your tool uses it
    }

    print(f"Running google_image_gen with prompt: {kwargs['prompt']}")

    # Run the google image gen tool
    # The decorated run function returns 5 elements
    result_str, input_prompt, metadata, callback, _api_keys_obj = run(**kwargs)

    print("\n--- Results ---")
    print(f"Input prompt: {input_prompt}")
    print(f"Metadata: {metadata}")
    print(f"Callback Data: {callback}")
    print(f"Result String: {result_str}")

    # Try parsing the result string if it's JSON
    try:
        result_data = json.loads(result_str)
        print("\nParsed Result Data:")
        print(f"  Image Hash: {result_data.get('image_hash')}")
        print(f"  Model Used: {result_data.get('model')}")
    except json.JSONDecodeError:
        print("\nResult is not valid JSON.")
    except Exception as e:
        print(f"\nCould not parse result string: {e}")

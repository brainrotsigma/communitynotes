import os

import dotenv
import requests
import base64


def _make_request(payload: dict):
    """
    Currently extremely simple and includes no retry logic.
    """
    chat_completions_url = "https://api.x.ai/v1/chat/completions"
    headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {os.getenv('XAI_API_KEY')}",
}
    response = requests.post(chat_completions_url, headers=headers, json=payload)
    if response.status_code != 200:
        raise Exception(f"Error making request: {response.status_code} {response.text}")
    return response.json()["choices"][0]["message"]["content"]


def get_grok_response(prompt: str, temperature: float = 0.8, model: str = "grok-3-latest"):
    payload = {
        "messages": [
            {
                "role": "system",
                "content": "You are a highly intelligent AI assistant.",
            },
            {"role": "user", "content": prompt},
        ],
        "model": model,
        "temperature": temperature,
    }
    return _make_request(payload)


def grok_describe_image(image_url: str, temperature: float = 0.01, model: str = "grok-2-vision-latest"):
    """
    Currently just describe image on its own. There are many possible
    improvements to consider making, e.g. passing in the post text or
    other context and describing the image and post text together.
    """
    payload = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": image_url,
                            "detail": "high",
                        },
                    },
                    {
                        "type": "text",
                        "text": "What's in this image?",
                    },
                ],
            },
        ],
        "model": model,
        "temperature": temperature,
    }
    return _make_request(payload)


def get_grok_live_search_response(prompt: str, temperature: float = 0.8, model= "grok-3-latest"):
    payload = {
        "messages": [{"role": "user", "content": prompt}],
        "search_parameters": {
            "mode": "on",
        },
        "model": model,
        "temperature": temperature,
    }
    return _make_request(payload)


def _gemini_generate_content(parts, temperature: float = 0.8, model: str = "gemini-2.5-flash"):
    """
    Minimal REST call for Gemini generateContent.
    Docs: https://ai.google.dev/api/generate-content
    """
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise Exception("GEMINI_API_KEY environment variable is not set.")
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
    headers = {"Content-Type": "application/json"}
    # The request payload mirrors the REST schema:
    # contents: [ { role: "user", parts: [ {text: ...} | {inlineData: ...} ] } ]
    # generationConfig: optional controls (e.g., temperature)
    payload = {
        "contents": [{"role": "user", "parts": parts}],
        "generationConfig": {"temperature": temperature},
    }
    response = requests.post(url, headers=headers, json=payload)
    if response.status_code != 200:
        raise Exception(f"Error making request to Gemini: {response.status_code} {response.text}")
    data = response.json()
    
    return data["candidates"][0]["content"]["parts"][0]["text"]


def get_gemini_response(prompt: str, temperature: float = 0.8, model: str = "gemini-2.5-flash"):
    parts = [{"text": prompt}]
    return _gemini_generate_content(parts, temperature, model)


def gemini_describe_image(image_url: str, temperature: float = 0.01, model: str = "gemini-2.5-flash"):
    """
    Describe the image at the provided URL using Gemini.
    """
    img_response = requests.get(image_url)
    if img_response.status_code != 200:
        raise Exception(f"Error fetching image: {img_response.status_code} {img_response.text}")

    mime_type = img_response.headers.get("Content-Type", "image/jpeg") # if Content-Type is not present, default to image/jpeg
    if ";" in mime_type:
        mime_type = mime_type.split(";")[0].strip()
    # Encode raw image bytes to base64 to satisfy the REST `inlineData` schema.
    image_b64 = base64.b64encode(img_response.content).decode("utf-8")
    parts = [
        {"text": "What's in this image?"},
        {"inlineData": {"mimeType": mime_type, "data": image_b64}},
    ]
    return _gemini_generate_content(parts, temperature, model)


def get_gemini_live_search_response(prompt: str, temperature: float = 0.8, model: str = "gemini-2.5-flash"):
    """
    Placeholder 'live search' using Gemini text generation.
    """
    parts = [{"text": prompt}]
    return _gemini_generate_content(parts, temperature, model)


if __name__ == "__main__":
    dotenv.load_dotenv()
    print(
        # get_grok_live_search_response(
        #     "Provide me a digest of world news in the last 2 hours. Please respond with links to each source next to the claims that the source supports."
        # )
        get_gemini_live_search_response(
            "Provide me a digest of world news in the last 2 hours. Please respond with links to each source next to the claims that the source supports."
        )
    )

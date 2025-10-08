import os
from typing import Any, Dict

import requests


def evaluate_note(
    note_text: str,
    post_id: str,
    verbose_if_failed: bool = False,
) -> Dict[str, Any]:
    """
    Call the Community Notes evaluate_note endpoint to get model-based
    evaluation for a proposed note.
    """
    token = os.environ.get("X_BEARER_TOKEN")
    if not token:
        raise RuntimeError("X_BEARER_TOKEN environment variable is not set.")

    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }
    payload = {"note_text": note_text, "post_id": post_id}
    resp = requests.post("https://api.x.com/2/evaluate_note", json=payload, headers=headers)
    resp.raise_for_status()
    return resp.json()



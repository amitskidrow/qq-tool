from __future__ import annotations

import json
import os
from typing import Dict

import google.generativeai as genai


def complete_json(model: str, system: str, user: str) -> Dict:
    key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not key:
        raise RuntimeError("GOOGLE_API_KEY/GEMINI_API_KEY missing")
    genai.configure(api_key=key)
    prompt = f"System:\n{system}\n\nUser:\n{user}\n\nReturn only JSON."
    m = genai.GenerativeModel(model)
    r = m.generate_content(prompt)
    txt = r.text
    # try to locate JSON object
    start = txt.find("{")
    end = txt.rfind("}")
    if start >= 0 and end > start:
        txt = txt[start : end + 1]
    return json.loads(txt)


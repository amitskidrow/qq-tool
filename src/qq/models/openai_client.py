from __future__ import annotations

import json
import os
from typing import Dict

from openai import OpenAI


def complete_json(model: str, system: str, user: str) -> Dict:
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY missing")
    client = OpenAI()
    # Try responses API first, fallback to chat.completions
    try:
        r = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0.2,
            response_format={"type": "json_object"},
        )
        content = r.choices[0].message.content
        return json.loads(content)
    except Exception as e:
        # As a fallback, try responses API
        try:
            rr = client.responses.create(
                model=model,
                input=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                temperature=0.2,
                response_format={"type": "json_object"},
            )
            txt = rr.output_text
            return json.loads(txt)
        except Exception as e2:
            raise


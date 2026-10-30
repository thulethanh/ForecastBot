from __future__ import annotations

import json
import re
from typing import Any, Dict


def extract_json(text: str) -> Dict[str, Any]:
    text = text.strip()

    # code fence
    code_block = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.DOTALL)
    if code_block:
        text = code_block.group(1)

    # first json object fallback
    if not text.startswith("{"):
        m = re.search(r"(\{.*\})", text, re.DOTALL)
        if m:
            text = m.group(1)

    return json.loads(text)
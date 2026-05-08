"""中英混合检索的轻量分词辅助。"""

from __future__ import annotations

import re


TOKEN_PATTERN = re.compile(r"[\u4e00-\u9fff]{2,}|[A-Za-z0-9][A-Za-z0-9_-]*")


def tokenize_mixed_text(text: str, *, deduplicate: bool = False) -> list[str]:
    if not isinstance(text, str):
        raise TypeError("tokenization input error: text must be string")

    tokens: list[str] = []
    for raw_token in TOKEN_PATTERN.findall(text):
        normalized = raw_token.lower().strip()
        if not normalized:
            continue
        if is_cjk_token(normalized):
            tokens.extend(_expand_cjk_token(normalized))
            continue
        tokens.append(normalized)
    return _deduplicate(tokens) if deduplicate else tokens


def is_cjk_token(token: str) -> bool:
    return bool(token) and all("\u4e00" <= char <= "\u9fff" for char in token)


def expand_cjk_query_token(token: str) -> list[str]:
    normalized = str(token).strip().lower()
    if not normalized:
        return []
    if not is_cjk_token(normalized):
        return [normalized]
    return _expand_cjk_token(normalized)


def _expand_cjk_token(token: str) -> list[str]:
    expanded = [token]
    if len(token) <= 2:
        return expanded
    expanded.extend(token[index : index + 2] for index in range(len(token) - 1))
    return expanded


def _deduplicate(tokens: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for token in tokens:
        if token in seen:
            continue
        ordered.append(token)
        seen.add(token)
    return ordered

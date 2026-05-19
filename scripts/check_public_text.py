from __future__ import annotations

import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SKIP_DIRS = {".git", ".local", ".venv", "__pycache__", "outputs", "checkpoints", "models", "logs", "runs"}
TEXT_SUFFIXES = {".md", ".py", ".yaml", ".yml", ".txt", ".json", ".jsonl", ".toml"}


def s(*codes: int) -> str:
    return "".join(chr(code) for code in codes)


BLOCKED = [
    s(35838, 31243),
    s(35838, 20214),
    s(32769, 24072),
    s(116, 101, 97, 99, 104, 101, 114),
    s(23578, 30821, 35895),
    s(35838, 22530),
    s(38754, 35797),
    s(65, 73, 32, 36741, 21161, 29983, 25104),
    s(83, 83, 72),
    s(112, 97, 115, 115, 119, 111, 114, 100),
]

PATTERNS = [
    re.compile(s(115, 115, 104) + r"\s+-p\s+\d+\s+\S+@\S+", re.IGNORECASE),
    re.compile("(" + "|".join([s(116, 111, 107, 101, 110), s(115, 101, 99, 114, 101, 116), s(112, 97, 115, 115, 119, 100)]) + r")\s*[:=]\s*\S+", re.IGNORECASE),
]


def iter_text_files():
    for path in ROOT.rglob("*"):
        if not path.is_file():
            continue
        if any(part in SKIP_DIRS for part in path.parts):
            continue
        if path.suffix.lower() in TEXT_SUFFIXES or path.name in {".gitignore"}:
            yield path


def main() -> int:
    hits: list[str] = []
    for path in iter_text_files():
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            text = path.read_text(encoding="utf-8", errors="ignore")
        rel = path.relative_to(ROOT)
        for term in BLOCKED:
            if term.lower() in text.lower():
                hits.append(f"{rel}: blocked wording")
        for pattern in PATTERNS:
            if pattern.search(text):
                hits.append(f"{rel}: sensitive pattern")

    if hits:
        print("\n".join(hits))
        return 1
    print("public text check passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())

from __future__ import annotations

import re
import sys
from pathlib import Path
from urllib.parse import unquote


ROOT = Path(__file__).resolve().parents[1]
LINK_RE = re.compile(r"(!?\[[^\]]*\]\(([^)]+)\))")


def iter_markdown_files():
    for path in [ROOT / "README.md", *sorted((ROOT / "docs").glob("*.md")), *sorted((ROOT / "reports").glob("*.md"))]:
        if path.exists():
            yield path


def is_external(target: str) -> bool:
    return "://" in target or target.startswith("#") or target.startswith("mailto:")


def strip_anchor(target: str) -> str:
    return target.split("#", 1)[0]


def main() -> int:
    errors: list[str] = []
    for md_path in iter_markdown_files():
        text = md_path.read_text(encoding="utf-8")
        for _, raw_target in LINK_RE.findall(text):
            target = unquote(raw_target.strip())
            if is_external(target):
                continue
            target = strip_anchor(target)
            if not target:
                continue
            resolved = (md_path.parent / target).resolve()
            if not resolved.exists():
                errors.append(f"{md_path.relative_to(ROOT)} -> {raw_target}")
    if errors:
        print("\n".join(errors))
        return 1
    print("markdown link check passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())

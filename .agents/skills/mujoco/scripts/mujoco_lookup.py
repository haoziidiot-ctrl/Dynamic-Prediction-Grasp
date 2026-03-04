#!/usr/bin/env python3
"""Quick local lookup for the MuJoCo skill references.

Usage:
  python .agents/skills/mujoco/scripts/mujoco_lookup.py --list
  python .agents/skills/mujoco/scripts/mujoco_lookup.py "joint axis"
  python .agents/skills/mujoco/scripts/mujoco_lookup.py "contact pair" --snippets 8
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, List, Tuple

ROOT = Path(__file__).resolve().parents[1]
REFS = ROOT / "references"

TOPICS: Dict[str, Dict[str, object]] = {
    "mjcf_core": {
        "keywords": ["mjcf", "xml", "body", "joint", "geom", "site", "default", "compiler", "option"],
        "url": "https://mujoco.readthedocs.io/en/latest/XMLreference.html",
        "files": ["01_mjcf_core.md"],
    },
    "controls_constraints": {
        "keywords": ["actuator", "sensor", "contact", "equality", "tendon", "pair", "rangefinder"],
        "url": "https://mujoco.readthedocs.io/en/latest/XMLreference.html",
        "files": ["02_elements_controls_constraints.md"],
    },
    "runtime_api": {
        "keywords": ["mj_step", "simulation", "qpos", "qvel", "ctrl", "viewer", "python", "data", "model"],
        "url": "https://mujoco.readthedocs.io/en/latest/programming/simulation.html",
        "files": ["03_runtime_python_c.md"],
    },
    "mjspec": {
        "keywords": ["mjspec", "model editing", "mjs_add", "compile", "recompile", "savexml"],
        "url": "https://mujoco.readthedocs.io/en/latest/programming/modeledit.html",
        "files": ["04_mjspec_model_editing.md"],
    },
    "debug_tuning": {
        "keywords": ["unstable", "diverge", "solver", "integrator", "debug", "collision", "memory"],
        "url": "https://mujoco.readthedocs.io/en/latest/modeling.html",
        "files": ["05_debugging_and_tuning.md"],
    },
    "recipes": {
        "keywords": ["template", "example", "snippet", "motor", "pd", "rangefinder"],
        "url": "https://mujoco.readthedocs.io/en/latest/",
        "files": ["06_templates_and_recipes.md"],
    },
}


def tokenize(text: str) -> List[str]:
    return [t for t in re.split(r"[^a-zA-Z0-9_]+", text.lower()) if t]


def score_topic(tokens: List[str], keywords: List[str]) -> int:
    keyset = set(k.lower() for k in keywords)
    score = 0
    for t in tokens:
        if t in keyset:
            score += 3
        else:
            score += sum(1 for k in keyset if t in k or k in t)
    return score


def list_topics() -> None:
    print("Available topics:")
    for name, meta in TOPICS.items():
        print(f"- {name}: {meta['url']}")
        print(f"  keywords: {', '.join(meta['keywords'])}")


def grep_snippets(query: str, max_lines: int = 6) -> List[Tuple[str, int, str]]:
    exact = re.compile(re.escape(query), re.IGNORECASE)
    tokens = [t for t in tokenize(query) if len(t) >= 2]
    token_patterns = [re.compile(re.escape(t), re.IGNORECASE) for t in tokens]

    exact_hits: List[Tuple[str, int, str]] = []
    token_hits: List[Tuple[int, str, int, str]] = []

    for path in sorted(REFS.glob("*.md")):
        with path.open("r", encoding="utf-8") as f:
            for idx, line in enumerate(f, start=1):
                txt = line.rstrip()
                if exact.search(txt):
                    exact_hits.append((path.name, idx, txt))
                    if len(exact_hits) >= max_lines:
                        return exact_hits
                    continue
                if token_patterns:
                    matched = sum(1 for p in token_patterns if p.search(txt))
                    if matched > 0:
                        token_hits.append((matched, path.name, idx, txt))

    if exact_hits:
        return exact_hits[:max_lines]

    token_hits.sort(key=lambda x: x[0], reverse=True)
    return [(fname, lineno, line) for _, fname, lineno, line in token_hits[:max_lines]]


def main() -> None:
    parser = argparse.ArgumentParser(description="Lookup MuJoCo skill references.")
    parser.add_argument("query", nargs="?", help="Search query, e.g. 'contact pair friction'.")
    parser.add_argument("--list", action="store_true", help="List indexed topics.")
    parser.add_argument("--snippets", type=int, default=6, help="Max snippet lines to print.")
    args = parser.parse_args()

    if args.list:
        list_topics()
        return

    if not args.query:
        parser.error("Provide a query or use --list.")

    tokens = tokenize(args.query)
    scored = []
    for name, meta in TOPICS.items():
        s = score_topic(tokens, meta["keywords"])  # type: ignore[arg-type]
        scored.append((s, name, meta))

    scored.sort(key=lambda x: x[0], reverse=True)

    print(f"Query: {args.query}")
    print("Top topics:")
    for s, name, meta in scored[:3]:
        print(f"- {name} (score={s})")
        print(f"  url: {meta['url']}")
        files = ", ".join(meta["files"])  # type: ignore[index]
        print(f"  files: {files}")

    print("\nMatching snippets:")
    snippets = grep_snippets(args.query, max_lines=args.snippets)
    if not snippets:
        print("- No direct line match. Try broader keywords or --list.")
        return

    for fname, lineno, line in snippets:
        print(f"- {fname}:{lineno}: {line}")


if __name__ == "__main__":
    main()

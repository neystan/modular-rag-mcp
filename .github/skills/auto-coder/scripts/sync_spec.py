#!/usr/bin/env python3
"""
Spec Sync — splits DEV_SPEC.md into chapter files under auto-coder/references/.

Usage:
    python scripts/sync_spec.py [--force]
"""

import hashlib
import re
import sys
from pathlib import Path
from typing import List, Tuple, NamedTuple


class Chapter(NamedTuple):
    number: int
    cn_title: str
    filename: str
    start_line: int
    end_line: int
    line_count: int


# Chapter number -> English slug (encoding-independent)
NUMBER_SLUG_MAP = {
    1: "overview",
    2: "features",
    3: "tech-stack",
    4: "testing",
    5: "architecture",
    6: "schedule",
    7: "future",
}


def _slug(chapter_num: int, title: str) -> str:
    if chapter_num in NUMBER_SLUG_MAP:
        return NUMBER_SLUG_MAP[chapter_num]
    # Fallback: sanitize whatever title text we have
    clean = re.sub(r'[^\w]+', '-', title, flags=re.ASCII).strip('-').lower()
    return clean or f"chapter-{chapter_num}"


def detect_chapters(content: str) -> List[Chapter]:
    lines = content.split('\n')
    starts: List[Tuple[int, str, int]] = []
    for i, line in enumerate(lines):
        m = re.match(r'^## (\d+)\.\s+(.+)$', line)
        if m:
            starts.append((int(m.group(1)), m.group(2).strip(), i))
    if not starts:
        raise ValueError("No chapters found. Expected '## N. Title'")
    chapters = []
    for idx, (num, title, start) in enumerate(starts):
        end = starts[idx + 1][2] if idx + 1 < len(starts) else len(lines)
        chapters.append(Chapter(num, title, f"{num:02d}-{_slug(num, title)}.md", start, end, end - start))
    return chapters


PHASE_KEYS = tuple("ABCDEFGHI")


def update_overall_progress(content: str) -> str:
    phase_stats = _collect_phase_stats(content)
    progress_table = _build_progress_table(phase_stats)
    pattern = re.compile(r"### 📈 总体进度\s*\n\s*\n\| 阶段 .*?\n\s*\n---", re.DOTALL)
    match = pattern.search(content)
    if not match:
        return content
    replacement = f"### 📈 总体进度\n\n{progress_table}\n\n---"
    return content[:match.start()] + replacement + content[match.end():]


def _collect_phase_stats(content: str) -> list[tuple[str, int, int]]:
    stats: list[tuple[str, int, int]] = []
    for phase_key in PHASE_KEYS:
        pattern = re.compile(
            rf"#### 阶段 {phase_key}：.*?\n\n\| 任务编号 .*?(?=\n#### 阶段 [A-I]：|\n---|\Z)",
            re.DOTALL,
        )
        match = pattern.search(content)
        if not match:
            continue

        phase_block = match.group(0)
        task_matches = re.findall(
            rf"^\|\s*{phase_key}[0-9.]+\s*\|.*?\|\s*\[([ x~])\]\s*\|",
            phase_block,
            re.MULTILINE,
        )
        total = len(task_matches)
        completed = sum(1 for status in task_matches if status == "x")
        stats.append((phase_key, total, completed))
    return stats


def _build_progress_table(phase_stats: list[tuple[str, int, int]]) -> str:
    lines = [
        "| 阶段 | 总任务数 | 已完成 | 进度 |",
        "|------|---------|--------|------|",
    ]

    total_tasks = 0
    total_completed = 0
    for phase_key, phase_total, phase_completed in phase_stats:
        total_tasks += phase_total
        total_completed += phase_completed
        lines.append(
            f"| 阶段 {phase_key} | {phase_total} | {phase_completed} | {_format_progress(phase_completed, phase_total)} |"
        )

    lines.append(
        f"| **总计** | **{total_tasks}** | **{total_completed}** | **{_format_progress(total_completed, total_tasks)}** |"
    )
    return "\n".join(lines)


def _format_progress(completed: int, total: int) -> str:
    if total <= 0:
        return "0%"
    return f"{round(completed / total * 100)}%"


def sync(force: bool = False):
    skill_dir = Path(__file__).parent.parent          # auto-coder/
    repo_root = skill_dir.parent.parent.parent        # project root
    dev_spec  = repo_root / "DEV_SPEC.md"
    specs_dir = skill_dir / "references"
    hash_file = skill_dir / ".spec_hash"

    if not dev_spec.exists():
        print(f"ERROR: {dev_spec} not found"); sys.exit(1)

    original_content = dev_spec.read_text(encoding='utf-8')
    normalized_content = update_overall_progress(original_content)
    if normalized_content != original_content:
        dev_spec.write_text(normalized_content, encoding='utf-8')

    # Hash check
    current_hash = hashlib.sha256(normalized_content.encode('utf-8')).hexdigest()
    if not force and hash_file.exists() and hash_file.read_text().strip() == current_hash:
        print("specs up-to-date"); return

    content = normalized_content
    chapters = detect_chapters(content)
    lines = content.split('\n')

    specs_dir.mkdir(parents=True, exist_ok=True)

    # Clean orphans
    old = {f.name for f in specs_dir.glob("*.md")}
    new = {ch.filename for ch in chapters}
    for f in old - new:
        (specs_dir / f).unlink()

    # Write chapters
    for ch in chapters:
        (specs_dir / ch.filename).write_text('\n'.join(lines[ch.start_line:ch.end_line]), encoding='utf-8')

    hash_file.write_text(current_hash)
    print(f"synced {len(chapters)} chapters")


if __name__ == "__main__":
    sync(force="--force" in sys.argv)

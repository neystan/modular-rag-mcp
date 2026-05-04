"""递归切分默认实现。"""

from __future__ import annotations

from libs.splitter.base_splitter import BaseSplitter


class RecursiveSplitter(BaseSplitter):
    """面向 Markdown 文本的默认递归切分器。"""

    def split_text(self, text: str, trace: object | None = None) -> list[str]:
        if not isinstance(text, str) or not text.strip():
            return []

        chunk_size = int(self.config.get("chunk_size", 1000))
        chunk_overlap = int(self.config.get("chunk_overlap", 200))
        if chunk_size <= 0:
            raise ValueError("recursive splitter config error: chunk_size must be positive")
        if chunk_overlap < 0:
            raise ValueError("recursive splitter config error: chunk_overlap must be non-negative")
        if chunk_overlap >= chunk_size:
            raise ValueError("recursive splitter config error: chunk_overlap must be smaller than chunk_size")

        chunks: list[str] = []
        for block in self._split_markdown_blocks(text):
            stripped_block = block.strip()
            if not stripped_block:
                continue
            if self._is_code_block(stripped_block):
                chunks.append(stripped_block)
                continue
            chunks.extend(self._split_prose_block(stripped_block, chunk_size, chunk_overlap))
        return chunks

    def _split_markdown_blocks(self, text: str) -> list[str]:
        blocks: list[str] = []
        current: list[str] = []
        in_code_block = False

        for line in text.splitlines(keepends=True):
            stripped = line.lstrip()
            if stripped.startswith("```"):
                if in_code_block:
                    current.append(line)
                    blocks.append("".join(current))
                    current = []
                    in_code_block = False
                    continue
                if current:
                    blocks.append("".join(current))
                    current = []
                current.append(line)
                in_code_block = True
                continue

            if in_code_block:
                current.append(line)
                continue

            if stripped.startswith("#") and current:
                blocks.append("".join(current))
                current = [line]
                continue

            current.append(line)

        if current:
            blocks.append("".join(current))
        return blocks

    @staticmethod
    def _is_code_block(block: str) -> bool:
        lines = block.splitlines()
        return bool(lines) and lines[0].lstrip().startswith("```")

    def _split_prose_block(self, text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
        if len(text) <= chunk_size:
            return [text]

        chunks: list[str] = []
        remaining = text
        separators = ["\n## ", "\n### ", "\n#### ", "\n\n", "\n", " ", ""]

        while len(remaining) > chunk_size:
            split_at = self._find_split_index(remaining, chunk_size, separators)
            chunk = remaining[:split_at].strip()
            if not chunk:
                split_at = chunk_size
                chunk = remaining[:split_at].strip()
            chunks.append(chunk)

            next_start = max(0, split_at - chunk_overlap)
            overlap_prefix = remaining[next_start:split_at]
            if chunk_overlap > 0 and overlap_prefix.strip():
                remaining = (overlap_prefix + remaining[split_at:]).strip()
            else:
                remaining = remaining[split_at:].strip()

        if remaining:
            chunks.append(remaining)
        return chunks

    def _find_split_index(self, text: str, chunk_size: int, separators: list[str]) -> int:
        for separator in separators:
            if not separator:
                continue
            index = text.rfind(separator, 0, chunk_size + 1)
            if index > 0:
                return index
        return chunk_size

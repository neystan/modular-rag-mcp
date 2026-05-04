"""Recursive Splitter 默认实现测试。"""

from __future__ import annotations

from core.settings import Settings
from libs.splitter.recursive_splitter import RecursiveSplitter
from libs.splitter.splitter_factory import SplitterFactory


def make_settings() -> Settings:
    return Settings(
        app={"name": "modular-rag-mcp"},
        llm={"provider": "placeholder"},
        embedding={"provider": "placeholder"},
        splitter={"provider": "recursive", "chunk_size": 80, "chunk_overlap": 10},
        vector_store={"provider": "placeholder"},
        retrieval={"top_k": 5},
        rerank={"provider": "none"},
        evaluation={"provider": "custom"},
        observability={"log_level": "INFO"},
    )


def test_factory_creates_recursive_splitter() -> None:
    splitter = SplitterFactory.create(make_settings())

    assert isinstance(splitter, RecursiveSplitter)


def test_split_text_preserves_markdown_headings_and_code_blocks() -> None:
    splitter = RecursiveSplitter({"chunk_size": 80, "chunk_overlap": 10})
    markdown = """# 标题一
第一段内容用于介绍背景。这一段会比较长，用来触发递归切分逻辑。

## 小节
这里继续补充说明，确保普通段落会被拆分。

```python
def hello():
    print("world")
```

收尾段落。"""

    chunks = splitter.split_text(markdown)

    assert chunks
    assert chunks[0].startswith("# 标题一")
    assert any(chunk.startswith("## 小节") for chunk in chunks)
    code_chunks = [chunk for chunk in chunks if chunk.startswith("```python")]
    assert code_chunks == ['```python\ndef hello():\n    print("world")\n```']
    assert any("收尾段落" in chunk for chunk in chunks)


def test_empty_text_returns_empty_list() -> None:
    splitter = RecursiveSplitter({"chunk_size": 80, "chunk_overlap": 10})

    assert splitter.split_text("") == []


def test_invalid_overlap_raises_readable_error() -> None:
    splitter = RecursiveSplitter({"chunk_size": 50, "chunk_overlap": 50})

    try:
        splitter.split_text("abc")
    except ValueError as exc:
        assert str(exc) == "recursive splitter config error: chunk_overlap must be smaller than chunk_size"
    else:
        raise AssertionError("expected ValueError")

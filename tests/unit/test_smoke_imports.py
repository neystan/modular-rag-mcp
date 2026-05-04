"""工程骨架冒烟测试。"""

from pathlib import Path


def test_top_level_packages_importable() -> None:
    import core
    import ingestion
    import libs
    import mcp_server
    import observability

    assert core is not None
    assert ingestion is not None
    assert libs is not None
    assert mcp_server is not None
    assert observability is not None


def test_prompt_templates_exist_and_are_readable() -> None:
    prompt_dir = Path("config/prompts")
    expected_files = {
        "image_captioning.txt",
        "chunk_refinement.txt",
        "rerank.txt",
    }

    for filename in expected_files:
        prompt_path = prompt_dir / filename
        assert prompt_path.is_file()
        assert prompt_path.read_text(encoding="utf-8").strip()

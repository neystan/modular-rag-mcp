"""运行黄金测试集评估。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from core.query_engine.hybrid_search import HybridSearch
from core.settings import load_settings
from libs.evaluator.evaluator_factory import EvaluatorFactory
from libs.llm.llm_factory import LLMFactory
from observability.evaluation.eval_runner import EvalRunner, EvalRunnerError


def build_parser() -> argparse.ArgumentParser:
    """构建命令行参数解析器。"""

    parser = argparse.ArgumentParser(description="运行 RAG 黄金测试集评估")
    parser.add_argument("--settings", default="config/settings.yaml", help="配置文件路径")
    parser.add_argument("--test-set", default="tests/fixtures/golden_test_set.json", help="黄金测试集 JSON 路径")
    return parser


def main(argv: list[str] | None = None) -> int:
    """脚本主入口。"""

    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        report = run_evaluation(settings_path=args.settings, test_set_path=args.test_set)
    except (ValueError, EvalRunnerError, ImportError) as exc:
        print(f"evaluate failed: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:  # noqa: BLE001
        print(f"evaluate failed: unexpected error: {exc}", file=sys.stderr)
        return 1

    print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2))
    return 0


def run_evaluation(
    *,
    settings_path: str | Path = "config/settings.yaml",
    test_set_path: str | Path = "tests/fixtures/golden_test_set.json",
) -> object:
    """创建默认组件并运行评估。"""

    settings = load_settings(settings_path)
    runner = EvalRunner(
        settings=settings,
        hybrid_search=HybridSearch(settings),
        evaluator=EvaluatorFactory.create(settings),
        llm=LLMFactory.create(settings),
    )
    return runner.run(test_set_path)


if __name__ == "__main__":
    raise SystemExit(main())

"""YAML 策略注册与动态加载。

策略通过 config/strategies.yaml 注册，每条目包含：
    name    策略名（用于输出标识）
    enabled 是否启用
    script  策略脚本相对路径
    params  策略参数（透传给策略构造器）

策略脚本需定义继承自 strategies.base.BaseStrategy 的 Strategy 类。
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def load_yaml(path: str | Path) -> dict[str, Any]:
    """读取 YAML 文件为 dict。"""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _resolve_script(script: str) -> Path:
    p = Path(script)
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    return p


def load_strategy_class(script: str):
    """从脚本文件动态加载 Strategy 类（不实例化）。"""
    script_path = _resolve_script(script)
    if not script_path.exists():
        raise FileNotFoundError(f"策略脚本不存在: {script_path}")

    # 以 strategies 包内模块名注册加载，使策略脚本中的相对导入
    # （from .base import BaseStrategy）可正常解析。
    import strategies  # noqa: F401  确保包已导入
    mod_name = f"strategies.{script_path.stem}"
    spec = importlib.util.spec_from_file_location(mod_name, script_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = module
    spec.loader.exec_module(module)

    cls = getattr(module, "Strategy", None)
    if cls is None:
        raise ValueError(f"策略脚本 {script_path} 未定义 Strategy 类")
    return cls


def load_strategies(strategies_yaml: str | Path) -> list:
    """加载并实例化所有启用的策略。

    返回 [(name, Strategy实例), ...]。
    """
    cfg = load_yaml(strategies_yaml)
    strategies: list = []
    for item in cfg.get("strategies", []):
        if not item.get("enabled", True):
            continue
        cls = load_strategy_class(item["script"])
        instance = cls(item.get("params") or {})
        strategies.append((item.get("name") or getattr(instance, "name", "unnamed"), instance))
    return strategies

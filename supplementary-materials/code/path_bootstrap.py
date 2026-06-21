import sys
from pathlib import Path


def _find_project_root(start: Path) -> Path:
    for p in (start, *start.parents):
        if (p / "experiments").exists() and (p / "huggingface_deploy").exists():
            return p
    return start.parent


def bootstrap() -> Path:
    root = _find_project_root(Path(__file__).resolve())

    candidates = [
        root,
        root / "02_模型代码",
        root / "04_评估工具",
        root / "02_模型代码" / "core" / "fusion",
        root / "02_模型代码" / "core" / "predict",
        root / "02_模型代码" / "core" / "loaders",
    ]

    for p in candidates:
        if p.exists():
            p_str = str(p)
            if p_str not in sys.path:
                sys.path.insert(0, p_str)

    return root

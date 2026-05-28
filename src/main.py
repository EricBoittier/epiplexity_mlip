from __future__ import annotations

from src.config import default_config
from src.experiment import run_experiments


def main() -> None:
    config = default_config()
    run_experiments(config)


if __name__ == "__main__":
    main()

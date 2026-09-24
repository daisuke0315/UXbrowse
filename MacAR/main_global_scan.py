#!/usr/bin/env python3
"""VS Code/CLI entry point for per-frame global-map PnP without tracking."""

from __future__ import annotations

import sys
from typing import Optional, Sequence

from realtime_pnp import main as realtime_main


def main(argv: Optional[Sequence[str]] = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    # Append last so this dedicated entry point cannot accidentally enable
    # temporal or reference-guided matching through inherited arguments.
    arguments.extend(("--matching-mode", "global"))
    return realtime_main(arguments)


if __name__ == "__main__":
    raise SystemExit(main())

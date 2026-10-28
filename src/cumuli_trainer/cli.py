# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Required Notice: Copyright 2026 Solipsist Studios Inc. (https://solipsist.studio)
"""``cumuli-train --config run.yaml [--resume ckpt.pt] [--key value ...]``

Any other ``--key value`` pair overrides the config, with dotted keys for
nested sections (``--strategy.cap_max 400000``). ``--list-plugins`` prints
every registered representation, strategy, loss and renderer.
"""

from __future__ import annotations

import argparse
import sys

from .config import ConfigError, load_config, parse_overrides


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    ap = argparse.ArgumentParser(prog="cumuli-train", description=__doc__.split("\n")[0])
    ap.add_argument("--config", help="run config (YAML or JSON)")
    ap.add_argument("--resume", help="checkpoint to continue from (ckpt_<iter>.pt)")
    ap.add_argument("--list-plugins", action="store_true")
    args, rest = ap.parse_known_args(argv)

    if args.list_plugins:
        from . import load_builtin_plugins, registry
        load_builtin_plugins()
        for kind in registry.KINDS:
            print(f"{kind}: {', '.join(registry.available(kind))}")
        return 0
    if not args.config:
        ap.error("--config is required")
    try:
        cfg = load_config(args.config, parse_overrides(rest))
    except ConfigError as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return 2

    from .trainer import Trainer
    Trainer(cfg, resume=args.resume).run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

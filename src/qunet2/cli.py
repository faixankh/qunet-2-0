from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(prog="qunet2")
    sub = parser.add_subparsers(dest="command")
    for name in ("train", "evaluate", "predict", "demo", "export"):
        sub.add_parser(name)
    args = parser.parse_args()
    print(f"QUNet 2.0 command: {args.command}")


if __name__ == "__main__":
    main()

from __future__ import annotations
import argparse
from pathlib import Path
from .config import Config
from .train import train
from .models.qunet2 import QUNet2
from .models.export import export_onnx
from .api.app import create_app

def main():
    parser = argparse.ArgumentParser(prog="qunet2")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_train = sub.add_parser("train")
    p_train.add_argument("--config", required=True)

    p_eval = sub.add_parser("evaluate")
    p_eval.add_argument("--config", required=True)

    p_export = sub.add_parser("export")
    p_export.add_argument("--output", default="outputs/qunet2.onnx")
    p_export.add_argument("--image-size", type=int, default=224)

    sub.add_parser("demo")

    args = parser.parse_args()
    if args.cmd in {"train", "evaluate"}:
        cfg = Config.load(args.config)
        result = train(cfg)
        print(result)
    elif args.cmd == "export":
        model = QUNet2()
        export_onnx(model, args.output, image_size=args.image_size)
        print(f"Exported to {args.output}")
    elif args.cmd == "demo":
        import uvicorn
        app = create_app()
        uvicorn.run(app, host="0.0.0.0", port=8000)
if __name__ == "__main__":
    main()

from __future__ import annotations

import importlib
import sys


MODULES = (
    "torch",
    "torchvision",
    "timm",
    "xformers",
    "pandas",
    "sklearn",
    "cv2",
    "PIL",
    "matplotlib",
    "albumentations",
    "medpy",
    "libauc",
)


def main() -> int:
    print(f"python: {sys.executable}")
    print(f"python_version: {sys.version.split()[0]}")

    failed = []
    for name in MODULES:
        try:
            module = importlib.import_module(name)
            version = getattr(module, "__version__", "installed")
            print(f"{name}: {version}")
        except Exception as exc:
            failed.append((name, str(exc)))
            print(f"{name}: FAILED ({exc})")

    try:
        import torch

        print(f"torch_cuda: {torch.version.cuda}")
        print(f"cuda_available: {torch.cuda.is_available()}")
        print(f"cuda_devices: {torch.cuda.device_count()}")
        if torch.cuda.is_available():
            print(f"gpu: {torch.cuda.get_device_name(0)}")
            x = torch.ones(1, device="cuda")
            print(f"cuda_smoke: {x.item():.1f}")
    except Exception as exc:
        failed.append(("torch_cuda", str(exc)))
        print(f"torch_cuda: FAILED ({exc})")

    if failed:
        print("\nEnvironment check failed.")
        return 1

    print("\nEnvironment check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

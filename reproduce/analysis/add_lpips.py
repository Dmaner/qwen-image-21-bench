"""Score C1 images with LPIPS against paired quality records."""

import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import platform
import statistics
from datetime import datetime, timezone
from pathlib import Path

import lpips
import numpy as np
import torch
from PIL import Image


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_image(path, expected):
    assert sha256(path) == expected["png_sha256"], f"PNG hash mismatch: {path}"
    with Image.open(path) as image:
        pixels = np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
    assert list(pixels.shape[1::-1]) == expected["size"], str(path)
    assert hashlib.sha256(pixels.tobytes()).hexdigest() == expected["rgb_sha256"], str(path)
    tensor = torch.from_numpy(pixels).permute(2, 0, 1).unsqueeze(0).float()
    return tensor / 127.5 - 1.0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics-root", type=Path, required=True)
    parser.add_argument("--images-root", type=Path, required=True,
                        help="Directory containing quality_bf16, quality_dit_fp8, etc.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--candidates", nargs="+", default=["dit_fp8", "te_fp8", "joint_fp8"])
    parser.add_argument("--threads", type=int, default=6)
    args = parser.parse_args()
    assert not args.output_dir.exists(), "Use a fresh output directory."
    assert len(set(args.candidates)) == len(args.candidates)
    torch.set_num_threads(args.threads)
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True)
    model = lpips.LPIPS(net="alex", version="0.1", lpips=True, pretrained=True,
                        pnet_rand=False, spatial=False, verbose=False).cpu().eval()
    weights = {
        "alexnet_imagenet": Path(torch.hub.get_dir()) / "checkpoints/alexnet-owt-7be5be79.pth",
        "lpips_calibration": Path(lpips.__file__).parent / "weights/v0.1/alex.pth",
    }
    result = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "CPU scoring of supplied C1 outputs against their paired quality records; no inference performed by this script.",
        "method": {
            "implementation": "https://github.com/richzhang/PerceptualSimilarity",
            "network": "alex", "lpips_version": "0.1", "pretrained": True,
            "learned_calibration": True, "spatial": False,
            "input": "Pillow RGB uint8 -> FP32 NCHW / 127.5 - 1; native resolution, no resize/crop",
            "device": "cpu", "dtype": "float32", "threads": args.threads,
            "aggregation": "Arithmetic mean of per-image distances over all 24 pairs per candidate",
            "lower_is_better": True,
        },
        "environment": {
            "python": platform.python_version(), "platform": platform.platform(),
            **{name: importlib.metadata.version(name)
               for name in ["lpips", "torch", "torchvision", "numpy", "Pillow", "scipy"]},
        },
        "weights": {key: {"filename": path.name, "sha256": sha256(path)}
                    for key, path in weights.items()},
        "script_sha256": sha256(Path(__file__)),
        "source_metrics_sha256": {}, "checks": {}, "pairs": [], "summary": {},
    }
    expected_ids = None
    with torch.inference_mode():
        for candidate in args.candidates:
            source = args.metrics_root / candidate / "quality_metrics.json"
            data = json.loads(source.read_text())
            assert data["reference_arm"] == "bf16" and data["candidate_arm"] == candidate
            pairs = data["pairs"]
            ids = [pair["case_id"] for pair in pairs]
            assert len(pairs) == len(set(ids)) == 24
            if expected_ids is None:
                expected_ids = ids
            assert ids == expected_ids, "Candidates must use the same ordered cases."
            result["source_metrics_sha256"][candidate] = sha256(source)
            scores = []
            for index, pair in enumerate(pairs):
                assert pair["status"] == "paired"
                case = pair["case_id"]
                reference = args.images_root / "quality_bf16" / "images" / f"{case}.png"
                tested = args.images_root / f"quality_{candidate}" / "images" / f"{case}.png"
                a = load_image(reference, pair["bf16"]["image"])
                b = load_image(tested, pair[candidate]["image"])
                assert a.shape == b.shape
                value = float(model(a, b).item())
                assert math.isfinite(value) and value >= 0, (candidate, case, value)
                if not result["checks"]:
                    identity = float(model(a, a).item())
                    reverse = float(model(b, a).item())
                    assert abs(identity) <= 1e-8
                    assert abs(value - reverse) <= 1e-7
                    result["checks"] = {
                        "case": case, "candidate": candidate,
                        "identity_distance": identity,
                        "symmetry_absolute_difference": abs(value - reverse),
                        "all_png_and_decoded_rgb_hashes_verified": True,
                    }
                scores.append(value)
                result["pairs"].append({
                    "candidate": candidate, "case_id": case,
                    "task": pair["task"], "split": pair["split"], "seed": pair["seed"],
                    "lpips": value,
                    "bf16_png_sha256": pair["bf16"]["image"]["png_sha256"],
                    "candidate_png_sha256": pair[candidate]["image"]["png_sha256"],
                })
                print(f"{candidate} {index + 1}/24 {case}: LPIPS={value:.8f}", flush=True)
            result["summary"][candidate] = {
                "n": len(scores), "mean_lpips": statistics.mean(scores),
                "median_lpips": statistics.median(scores),
                "min_lpips": min(scores), "max_lpips": max(scores),
                "splits": {
                    split: {"n": sum(pair["split"] == split for pair in pairs),
                            "mean_lpips": statistics.mean(value for pair, value in zip(pairs, scores)
                                                          if pair["split"] == split)}
                    for split in sorted({pair["split"] for pair in pairs})
                },
            }
    args.output_dir.mkdir(parents=True)
    (args.output_dir / "lpips_metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    with (args.output_dir / "lpips_pairs.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(result["pairs"][0]))
        writer.writeheader()
        writer.writerows(result["pairs"])
    print(json.dumps(result["summary"], indent=2), flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""MuseTalk CLI wrapper for the GuiAssert-MuseTalk plugin.

Invokes the upstream `scripts/inference.py` against a portrait image
(or short video) + WAV narration, and produces a single MP4 at the
requested output path.

Used by GuiAssert's `talking_head` module via subprocess. Exits 0 on
success, non-zero with a diagnostic message on failure.

Usage:
    python render_musetalk.py \\
        --audio /path/to/narration.wav \\
        --source-image /path/to/portrait.png \\
        --output /path/to/musetalk.mp4 \\
        [--device mps|cpu|cuda|auto] [--version v15|v1]

Design choices:
  * `--device auto` picks MPS if available, otherwise CPU. MuseTalk's
    upstream `scripts/inference.py` only knows about `cuda` / `cpu`;
    we patch it to honour `MUSETALK_DEVICE_OVERRIDE` (see PATCHES.md)
    so we can force `mps`. The `PYTORCH_ENABLE_MPS_FALLBACK=1` env var
    (set below) covers any ops MPS does not implement.
  * MuseTalk's inference is config-driven: the upstream script reads an
    OmegaConf YAML and iterates over its top-level keys. We generate
    that YAML on the fly from the `--audio` / `--source-image` /
    `--output` arguments, so callers see a flat CLI.
  * MuseTalk writes intermediate state into `<result_dir>/<version>/`.
    We point that at a per-call temp directory and move the produced
    MP4 to the caller-supplied `--output` path on success.
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path


def resolve_device(requested: str) -> str:
    """Pick the actual device string."""
    if requested == "cpu":
        return "cpu"
    if requested == "mps":
        return "mps"
    if requested == "cuda":
        return "cuda"
    # auto
    try:
        import torch  # local import — keeps the script importable for --help
        if torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


def main() -> int:
    parser = argparse.ArgumentParser(description="MuseTalk CLI wrapper.")
    parser.add_argument("--audio", required=True, help="Narration WAV path")
    parser.add_argument("--source-image", required=True,
                        help="Portrait PNG/JPG (or a short video file)")
    parser.add_argument("--output", required=True, help="Destination MP4 path")
    parser.add_argument("--device", default="auto",
                        choices=["auto", "mps", "cpu", "cuda"])
    parser.add_argument("--version", default="v15",
                        choices=["v15", "v1"],
                        help="Which MuseTalk model version to use.")
    parser.add_argument("--fps", type=int, default=25,
                        help="Output video FPS (applies when source is an image).")
    parser.add_argument("--batch-size", type=int, default=4,
                        help="Inference batch size. Lower for low-VRAM / MPS.")
    parser.add_argument("--bbox-shift", type=int, default=0,
                        help="Bounding-box shift used by v1 (v15 ignores this).")
    parser.add_argument("--use-float16", action="store_true",
                        help="Use FP16 weights (CUDA only; on MPS this is a no-op).")
    args = parser.parse_args()

    audio = Path(args.audio).resolve()
    source = Path(args.source_image).resolve()
    output = Path(args.output).resolve()

    if not audio.exists():
        print(f"ERROR: audio not found: {audio}", file=sys.stderr)
        return 2
    if not source.exists():
        print(f"ERROR: source image not found: {source}", file=sys.stderr)
        return 2
    output.parent.mkdir(parents=True, exist_ok=True)

    # The wrapper lives at python/render_musetalk.py inside the
    # GuiAssert-MuseTalk checkout; MuseTalk upstream is the sibling
    # `upstream/` folder.
    here = Path(__file__).resolve().parent
    upstream = here / "upstream"
    inference_module = "scripts.inference"
    models = upstream / "models"

    if not (upstream / "scripts" / "inference.py").exists():
        print(f"ERROR: MuseTalk upstream not found at {upstream}",
              file=sys.stderr)
        return 3

    # Required weight files for the selected version.
    if args.version == "v15":
        unet_model = models / "musetalkV15" / "unet.pth"
        unet_config = models / "musetalkV15" / "musetalk.json"
    else:
        unet_model = models / "musetalk" / "pytorch_model.bin"
        unet_config = models / "musetalk" / "musetalk.json"

    for required in (
        unet_model, unet_config,
        models / "sd-vae" / "diffusion_pytorch_model.bin",
        models / "whisper" / "pytorch_model.bin",
        models / "dwpose" / "dw-ll_ucoco_384.pth",
    ):
        if not required.exists():
            print(f"ERROR: required weight missing: {required}",
                  file=sys.stderr)
            return 3

    device = resolve_device(args.device)
    print(f"[render_musetalk] device={device} version={args.version} "
          f"batch_size={args.batch_size} fps={args.fps}")

    # Generate the inference YAML on the fly.
    with tempfile.TemporaryDirectory(prefix="musetalk-run-") as tmp_dir:
        tmp = Path(tmp_dir)
        config_path = tmp / "inference.yaml"
        result_dir = tmp / "results"
        result_dir.mkdir(parents=True, exist_ok=True)

        # The output_vid_name flag controls only the leaf filename; the
        # MP4 ends up under `<result_dir>/<version>/<output_vid_name>`.
        out_leaf = "out.mp4"

        # OmegaConf accepts YAML; we hand-write the minimal task block
        # documented by upstream's configs/inference/test.yaml.
        config_yaml = (
            "task_0:\n"
            f' video_path: "{source.as_posix()}"\n'
            f' audio_path: "{audio.as_posix()}"\n'
        )
        if args.version == "v1":
            config_yaml += f' bbox_shift: {args.bbox_shift}\n'
        config_path.write_text(config_yaml)

        cmd = [
            sys.executable, "-m", inference_module,
            "--inference_config", str(config_path),
            "--result_dir", str(result_dir),
            "--unet_model_path", str(unet_model),
            "--unet_config", str(unet_config),
            "--whisper_dir", str(models / "whisper"),
            "--version", args.version,
            "--batch_size", str(args.batch_size),
            "--fps", str(args.fps),
            "--output_vid_name", out_leaf,
        ]
        if args.use_float16:
            cmd.append("--use_float16")

        env = os.environ.copy()
        # Once weights are local, no further HF / transformers downloads
        # should fire at inference time.
        env.setdefault("HF_HUB_OFFLINE", "1")
        env.setdefault("TRANSFORMERS_OFFLINE", "1")
        env.setdefault("PYTHONUNBUFFERED", "1")
        # Apple Silicon MPS — fall back to CPU on ops the MPS backend
        # doesn't implement. Without this PyTorch raises
        # NotImplementedError for several aten kernels MuseTalk touches.
        env.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
        # Patch (1) in PATCHES.md teaches upstream's inference.py and
        # the preprocessing init-time face-alignment / mmpose calls to
        # honour this env var. Without it, upstream picks cuda / cpu.
        env["MUSETALK_DEVICE_OVERRIDE"] = device

        print(f"[render_musetalk] running: {' '.join(cmd)}")
        started = time.time()
        try:
            proc = subprocess.run(
                cmd, cwd=str(upstream), env=env, check=False)
        except FileNotFoundError as e:
            print(f"ERROR: failed to invoke python: {e}", file=sys.stderr)
            return 4
        elapsed = time.time() - started
        print(f"[render_musetalk] musetalk exit={proc.returncode} "
              f"elapsed={elapsed:.1f}s")
        if proc.returncode != 0:
            print(f"ERROR: MuseTalk inference failed with exit code "
                  f"{proc.returncode}", file=sys.stderr)
            return proc.returncode

        # Find the produced MP4 under result_dir/<version>/.
        produced = result_dir / args.version / out_leaf
        if not produced.exists():
            # Upstream sometimes nests differently; scan one level down.
            candidates = sorted(result_dir.rglob("*.mp4"))
            if not candidates:
                print(f"ERROR: no MP4 produced under {result_dir}",
                      file=sys.stderr)
                return 5
            produced = candidates[-1]
        if produced.stat().st_size == 0:
            print(f"ERROR: produced MP4 is empty: {produced}",
                  file=sys.stderr)
            return 5

        # Move into place.
        if output.exists():
            output.unlink()
        shutil.move(str(produced), str(output))

    print(f"[render_musetalk] produced: {output} "
          f"({output.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

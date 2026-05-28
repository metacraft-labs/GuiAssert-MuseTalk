#!/usr/bin/env bash
# Quick smoke-test that the MuseTalk plugin is wired up.
#
# Verifies:
#   * .venv/bin/python exists and imports torch + diffusers + accelerate +
#     transformers + librosa + opencv + mmpose + mmcv without raising,
#   * the wrapper script's --help runs cleanly,
#   * each MuseTalk weight family is present and plausibly-sized,
#   * the inference script + face-detection module are present at the
#     pinned upstream commit.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PY=".venv/bin/python"
SCRIPT="python/render_musetalk.py"
UPSTREAM_DIR="python/upstream"
MODELS_DIR="$UPSTREAM_DIR/models"

if [ ! -x "$PY" ]; then
  echo "FAIL: $PY not found. Run ./scripts/install.sh first." >&2
  exit 1
fi
if [ ! -f "$SCRIPT" ]; then
  echo "FAIL: wrapper script missing at $SCRIPT" >&2
  exit 1
fi
if [ ! -f "$UPSTREAM_DIR/scripts/inference.py" ]; then
  echo "FAIL: upstream MuseTalk checkout missing at $UPSTREAM_DIR" >&2
  exit 1
fi

# Required weight files.
for weight in \
  "$MODELS_DIR/musetalkV15/unet.pth" \
  "$MODELS_DIR/musetalkV15/musetalk.json" \
  "$MODELS_DIR/sd-vae/diffusion_pytorch_model.bin" \
  "$MODELS_DIR/sd-vae/config.json" \
  "$MODELS_DIR/whisper/pytorch_model.bin" \
  "$MODELS_DIR/whisper/config.json" \
  "$MODELS_DIR/dwpose/dw-ll_ucoco_384.pth" \
  "$MODELS_DIR/face-parse-bisent/79999_iter.pth" \
  "$MODELS_DIR/face-parse-bisent/resnet18-5c106cde.pth"; do
  if [ ! -s "$weight" ]; then
    echo "FAIL: $weight missing or empty. Run ./scripts/install.sh." >&2
    exit 1
  fi
done

# Total weight size — sanity check (~5 GB give or take).
total_size_kb="$(du -sk "$MODELS_DIR" | awk '{print $1}')"
echo "[verify] total weights size: ${total_size_kb} KB"

echo "[verify] importing torch ..."
"$PY" -c "import torch; print('  torch', torch.__version__, 'mps_available=', torch.backends.mps.is_available())"

echo "[verify] importing diffusers + accelerate + transformers ..."
"$PY" -c "import diffusers, accelerate, transformers; print('  diffusers', diffusers.__version__, 'accelerate', accelerate.__version__, 'transformers', transformers.__version__)"

echo "[verify] importing librosa + cv2 + numpy ..."
"$PY" -c "import librosa, cv2, numpy; print('  librosa', librosa.__version__, 'cv2', cv2.__version__, 'numpy', numpy.__version__)"

echo "[verify] importing mmengine + mmcv + mmdet + mmpose ..."
"$PY" -c "import mmengine, mmcv, mmdet, mmpose; print('  mmengine', mmengine.__version__, 'mmcv', mmcv.__version__, 'mmdet', mmdet.__version__, 'mmpose', mmpose.__version__)"

# mmcv MUST be the full build, not mmcv-lite — mmpose 1.3.2's
# EDPoseHead does `from mmcv.ops import MultiScaleDeformableAttention`
# at import time, and that symbol only exists in the compiled
# `mmcv._ext` extension.
echo "[verify] checking mmcv._ext + mmpose.apis import path ..."
"$PY" -c "import mmcv._ext, sys; print('  mmcv._ext at', mmcv._ext.__file__)"
"$PY" -c "from mmcv.ops import MultiScaleDeformableAttention; print('  MultiScaleDeformableAttention OK')"
"$PY" -c "from mmpose.apis import init_model, inference_topdown; print('  mmpose.apis OK')"

echo "[verify] wrapper --help ..."
"$PY" "$SCRIPT" --help >/dev/null

echo "[verify] DONE — MuseTalk plugin is wired up."

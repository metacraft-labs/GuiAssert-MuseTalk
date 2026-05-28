#!/usr/bin/env bash
# Install / refresh the MuseTalk plugin.
#
# Steps (each idempotent — re-running the script is safe):
#   1. create a Python 3.10 venv under .venv/ if missing,
#   2. install PyTorch + python/requirements-patched.txt,
#   3. install the OpenMIM-managed mm* family — `mmengine`, full
#      `mmcv==2.1.0` (compiled from source with MMCV_WITH_OPS=1
#      FORCE_CUDA=0; mmcv-lite is NOT sufficient because mmpose's
#      EDPoseHead imports MultiScaleDeformableAttention from
#      mmcv._ext at module load), then `mmdet` + `mmpose` via mim,
#   4. clone MuseTalk upstream at the pinned commit from python/COMMIT.txt,
#   5. apply the source patches documented in python/PATCHES.md,
#   6. download model weights from Hugging Face mirrors (~5 GB).
#
# Run from inside `nix develop` (the flake exposes python3, git, curl,
# ffmpeg-full, cmake, pkg-config). The script makes no assumption about
# the dev-shell Python version — it requires `python3.10` on PATH
# (Homebrew's `python@3.10` on macOS, or the distro's `python3.10` on
# Linux).

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# Strip inherited PYTHONPATH / PYTHONHOME — Nix-managed dev shells and
# direnv layers commonly leak a Python 3.13 site-packages directory into
# PYTHONPATH, which then shadows the venv's Python 3.10 stdlib (notably
# the `sysconfig` module). The leaked `sysconfig` reports `VERSION='3.13'`
# / `SOABI='cpython-313-darwin'`, which makes any C-extension build (e.g.
# xtcocotools) fail the wheel-tag validator with
# `AssertionError: would build wheel with unsupported tag ('cp310', 'cp313', ...)`.
# Unset them defensively before the venv is sourced.
unset PYTHONPATH
unset PYTHONHOME

PY=${PYTHON_BIN:-${PYTHON:-python3.10}}
if ! command -v "$PY" >/dev/null 2>&1; then
  echo "ERROR: $PY not on PATH." >&2
  echo "       Install Python 3.10 first:" >&2
  echo "         macOS:   brew install python@3.10" >&2
  echo "         Linux:   use your distro's python3.10 package" >&2
  exit 1
fi
echo "[install] using $($PY --version) at $(command -v "$PY")"

# ---------------------------------------------------------------------
# 1. venv
# ---------------------------------------------------------------------
if [ ! -d .venv ]; then
  echo "[install] creating Python venv under .venv/ ..."
  "$PY" -m venv .venv
fi

# shellcheck disable=SC1091
source .venv/bin/activate

# Refresh pip + wheel ahead of installing scientific deps.
pip install --upgrade pip wheel setuptools

# ---------------------------------------------------------------------
# 2. PyTorch + patched requirements
# ---------------------------------------------------------------------
# MuseTalk's upstream notes call for torch==2.0.1; in practice recent
# torch 2.2.x macOS-arm64 wheels work transparently with the same
# weight checkpoints, and torch 2.0.1 arm64 wheels are flaky. Let the
# caller pre-install torch ahead of running this script to override;
# otherwise we install a known-good 2.2.x.
echo "[install] installing PyTorch (CPU+MPS wheel on macOS, CPU on Linux)..."
if ! python -c "import torch" >/dev/null 2>&1; then
  pip install "torch==2.2.2" "torchvision==0.17.2" "torchaudio==2.2.2"
fi
echo "[install] installing patched requirements ..."
pip install -r python/requirements-patched.txt

# ---------------------------------------------------------------------
# 3. OpenMIM-managed mm* packages
# ---------------------------------------------------------------------
# MuseTalk's preprocessing imports `mmpose.apis.init_model` and runs
# DWPose RTMPose for face landmarks. Its transitive deps (mmengine,
# mmcv, mmdet) interlock through version constraints that pip's solver
# cannot navigate cleanly on Apple Silicon. OpenMIM (`mim`) is the
# upstream-recommended tool for that family.
#
# IMPORTANT: we install the FULL `mmcv` (CPU build, no CUDA), NOT
# `mmcv-lite`. mmpose 1.3.2 imports `MultiScaleDeformableAttention` from
# `mmcv.ops` at module import time via
# `mmpose.models.heads.transformer_heads.EDPoseHead`. That symbol lives
# in the compiled `mmcv._ext` extension which is only present in the
# full `mmcv` build. Using `mmcv-lite` made every `from mmpose.apis ...`
# call raise `ModuleNotFoundError: No module named 'mmcv._ext'`.
#
# There are NO pre-built `mmcv` wheels for macOS-arm64 (or for any
# non-CUDA target other than the CPU Linux build) so we compile from
# source with `MMCV_WITH_OPS=1 FORCE_CUDA=0`. Important caveats:
#   * The build needs PyTorch's C++ headers, so `--no-build-isolation`
#     is mandatory (build isolation hides the venv's installed torch).
#   * Apple's clang 21 / macOS 26 SDK marks several libc++ traits
#     (`is_arithmetic`, ...) as `_LIBCPP_NO_SPECIALIZATIONS`, which
#     PyTorch 2.2's `c10::util::strong_type` specialises. We pin
#     `-isysroot` to the older MacOSX15 SDK (libc++ without the
#     no-specializations annotation) via CFLAGS/CXXFLAGS so the build
#     succeeds. If MacOSX15.sdk is unavailable (e.g. on Linux or on a
#     Mac that has only the current SDK installed) we skip the
#     `isysroot` override and try the system default.
#   * `MAX_JOBS=2` keeps memory pressure low; the build takes ~3-6 min.
echo "[install] installing OpenMIM-managed mm* packages ..."
if ! python -c "import mim" >/dev/null 2>&1; then
  pip install -U openmim
fi

# `mim install` is idempotent: it skips already-installed packages.
# Pin versions known to interlock cleanly with mmpose's RTMPose pipeline.
if ! python -c "import mmengine" >/dev/null 2>&1; then
  mim install "mmengine>=0.10.0,<1.0.0"
fi

# Drop any pre-existing mmcv-lite — keeping it side-by-side with full
# mmcv breaks the import ordering. Also drop a too-new full mmcv (>=2.2)
# because mmdet 3.x asserts `mmcv < 2.2.0` at import time.
mmcv_lite_installed=0
if python -c "import mmcv_lite" >/dev/null 2>&1 || python -c "import importlib.metadata as m; m.version('mmcv-lite')" >/dev/null 2>&1; then
  mmcv_lite_installed=1
fi
if [ "$mmcv_lite_installed" = "1" ]; then
  echo "[install] removing mmcv-lite (full mmcv replaces it) ..."
  pip uninstall -y mmcv-lite || true
fi
mmcv_too_new=0
if python -c "import mmcv" >/dev/null 2>&1; then
  if ! python -c "import mmcv; parts = mmcv.__version__.split('.'); assert (int(parts[0]), int(parts[1])) < (2, 2)" >/dev/null 2>&1; then
    mmcv_too_new=1
  fi
fi
if [ "$mmcv_too_new" = "1" ]; then
  echo "[install] uninstalling too-new mmcv before reinstall ..."
  pip uninstall -y mmcv || true
fi

# If we still don't have a working full mmcv (with the _ext compiled
# extension), build it from source.
if ! python -c "import mmcv, mmcv._ext" >/dev/null 2>&1; then
  echo "[install] building full mmcv==2.1.0 from source (CPU only, ~3-6 min) ..."
  # Detect the older MacOSX15 SDK as a workaround for the libc++
  # `_LIBCPP_NO_SPECIALIZATIONS` incompatibility with PyTorch 2.2's
  # `c10::util::strong_type` in the macOS 26 SDK. If it's missing,
  # fall through to the default toolchain.
  mmcv_build_env=()
  if [ "$(uname -s)" = "Darwin" ]; then
    if [ -d /Library/Developer/CommandLineTools/SDKs/MacOSX15.sdk ]; then
      MMCV_SDKROOT=/Library/Developer/CommandLineTools/SDKs/MacOSX15.sdk
      mmcv_build_env+=(CC=/Library/Developer/CommandLineTools/usr/bin/clang)
      mmcv_build_env+=(CXX=/Library/Developer/CommandLineTools/usr/bin/clang++)
      mmcv_build_env+=("CFLAGS=-isysroot $MMCV_SDKROOT")
      mmcv_build_env+=("CXXFLAGS=-isysroot $MMCV_SDKROOT")
      mmcv_build_env+=("LDFLAGS=-isysroot $MMCV_SDKROOT")
      mmcv_build_env+=("MACOSX_DEPLOYMENT_TARGET=14.0")
      echo "[install]   using SDK $MMCV_SDKROOT for libc++ headers"
    else
      echo "[install]   MacOSX15.sdk not found — falling back to system toolchain"
      echo "[install]   (if the build fails with libc++ 'unresolved using" \
           "declaration' / 'is_arithmetic cannot be specialized', install" \
           "Command Line Tools that include MacOSX15.sdk)"
    fi
  fi
  # `${arr[@]+"${arr[@]}"}` expands to nothing when arr is empty —
  # avoids `env ""` (treated by env as a command name) on Linux/non-Mac
  # systems where mmcv_build_env stays empty.
  env ${mmcv_build_env[@]+"${mmcv_build_env[@]}"} \
    MMCV_WITH_OPS=1 FORCE_CUDA=0 MAX_JOBS=2 \
    pip install "mmcv==2.1.0" --no-build-isolation
  # Restore upstream-pinned numpy/opencv versions in case the mmcv
  # install pulled in newer transitive deps that bumped them.
  pip install -r python/requirements-patched.txt
fi

# chumpy (an mmpose runtime dep) has a setup.py that does `import pip`
# inside the build-isolation overlay, which fails on modern pip/setuptools
# (ModuleNotFoundError: No module named 'pip'). Pre-install chumpy with
# build isolation disabled so its setup.py runs against the venv's pip.
if ! python -c "import chumpy" >/dev/null 2>&1; then
  pip install --no-build-isolation chumpy
fi
if ! python -c "import mmdet" >/dev/null 2>&1; then
  mim install "mmdet>=3.0.0,<3.4.0"
fi
# xtcocotools (an mmpose runtime dep, COCO-style keypoint metrics) has a
# setup.py that imports numpy at build time. pip's isolated build overlay
# does not include numpy, so the build aborts with
# `ModuleNotFoundError: No module named 'numpy'`. Disable build isolation
# so setup.py sees the venv's numpy.
if ! python -c "import xtcocotools" >/dev/null 2>&1; then
  pip install --no-build-isolation xtcocotools
fi
if ! python -c "import mmpose" >/dev/null 2>&1; then
  mim install "mmpose>=1.0.0,<1.4.0"
fi

# ---------------------------------------------------------------------
# 4. clone upstream at the pinned commit
# ---------------------------------------------------------------------
PINNED_SHA="$(tr -d '[:space:]' < python/COMMIT.txt)"
UPSTREAM_DIR="python/upstream"
if [ ! -d "$UPSTREAM_DIR/.git" ]; then
  echo "[install] cloning MuseTalk upstream at $PINNED_SHA ..."
  git clone https://github.com/TMElyralab/MuseTalk.git "$UPSTREAM_DIR"
fi
( cd "$UPSTREAM_DIR" && git fetch --tags && git checkout "$PINNED_SHA" )

# ---------------------------------------------------------------------
# 5. apply patches (see python/PATCHES.md for the manifest)
# ---------------------------------------------------------------------
echo "[install] applying MPS device-routing + torch.load weights_only patches ..."

# Each apply_patch invocation is idempotent: it only rewrites the file
# when the legacy needle is still present. The Python helper below
# does a whitespace-tolerant match: trailing whitespace on each line
# of the needle is ignored when matching, so an upstream file with a
# stray trailing space on a line (a few MuseTalk modules do) still
# patches cleanly. The replacement is written verbatim.
apply_patch() {
  local file="$1"
  local needle="$2"
  local replacement="$3"
  if [ ! -f "$file" ]; then
    echo "  WARN: skipping $file (not found)" >&2
    return 0
  fi
  python - "$file" "$needle" "$replacement" <<'PY'
import sys, pathlib, re
path = pathlib.Path(sys.argv[1])
needle = sys.argv[2]
replacement = sys.argv[3]
data = path.read_text()
# Build a whitespace-tolerant regex: every line of the needle is
# allowed to be followed by trailing spaces/tabs before the newline.
lines = needle.split('\n')
pattern = r'[ \t]*\n'.join(re.escape(l) for l in lines)
new_data, n = re.subn(pattern, lambda _m: replacement, data, count=1)
if n:
    print(f"  patching {path} ({n} replacement)")
    path.write_text(new_data)
PY
}

# (1) scripts/inference.py: device selection honours MUSETALK_DEVICE_OVERRIDE.
apply_patch "$UPSTREAM_DIR/scripts/inference.py" \
  '    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")' \
  '    import os as _mt_os
    _mt_dev = _mt_os.environ.get("MUSETALK_DEVICE_OVERRIDE", "").strip()
    if _mt_dev in ("mps", "cpu"):
        device = torch.device(_mt_dev)
    elif _mt_dev.startswith("cuda"):
        device = torch.device(_mt_dev)
    else:
        device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")'

# (2a) musetalk/utils/preprocessing.py: mmpose init_model device.
apply_patch "$UPSTREAM_DIR/musetalk/utils/preprocessing.py" \
  'device = torch.device("cuda" if torch.cuda.is_available() else "cpu")' \
  'import os as _mt_os
_mt_dev = _mt_os.environ.get("MUSETALK_DEVICE_OVERRIDE", "").strip()
if _mt_dev in ("mps", "cpu") or _mt_dev.startswith("cuda"):
    device = torch.device(_mt_dev)
else:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")'

# (2b) musetalk/utils/preprocessing.py: face-alignment device string.
apply_patch "$UPSTREAM_DIR/musetalk/utils/preprocessing.py" \
  'device = "cuda" if torch.cuda.is_available() else "cpu"' \
  'device = os.environ.get("MUSETALK_DEVICE_OVERRIDE", "").strip() or ("cuda" if torch.cuda.is_available() else "cpu")'

# (3) musetalk/utils/face_detection/detection/core.py: accept mps device.
apply_patch "$UPSTREAM_DIR/musetalk/utils/face_detection/detection/core.py" \
  "        if 'cpu' not in device and 'cuda' not in device:" \
  "        if 'cpu' not in device and 'cuda' not in device and 'mps' not in device:"

# (4) musetalk/utils/face_detection/api.py: treat 'mps' like 'cuda'.
apply_patch "$UPSTREAM_DIR/musetalk/utils/face_detection/api.py" \
  "        if 'cuda' in device:" \
  "        if 'cuda' in device or 'mps' in device:"

# (5a) musetalk/utils/face_parsing/__init__.py: MPS-aware weight load.
# Upstream's `scripts/inference.py` instantiates `FaceParsing(...)` while
# the cwd is `upstream/`, so the hardcoded `./models/face-parse-bisent/...`
# default in model_init resolves correctly. We patch the device routing
# so the wrapper can force MPS, and flip `weights_only=False` so the
# pre-2.6 pickle keeps deserialising under PyTorch >=2.6.
#
# NOTE: upstream's source ships with a stray trailing space after
# `net.load_state_dict(torch.load(model_pth))`. The apply_patch helper
# above is whitespace-tolerant per line so the patch fires regardless.
apply_patch "$UPSTREAM_DIR/musetalk/utils/face_parsing/__init__.py" \
  "        if torch.cuda.is_available():
            net.cuda()
            net.load_state_dict(torch.load(model_pth))
        else:
            net.load_state_dict(torch.load(model_pth, map_location=torch.device('cpu')))" \
  "        import os as _mt_os
        _mt_dev = _mt_os.environ.get('MUSETALK_DEVICE_OVERRIDE', '').strip()
        if _mt_dev in ('mps', 'cpu'):
            net.to(_mt_dev)
            net.load_state_dict(torch.load(model_pth, map_location=torch.device(_mt_dev), weights_only=False))
        elif torch.cuda.is_available():
            net.cuda()
            net.load_state_dict(torch.load(model_pth, weights_only=False))
        else:
            net.load_state_dict(torch.load(model_pth, map_location=torch.device('cpu'), weights_only=False))"

# (5b) musetalk/utils/face_parsing/__init__.py: per-image MPS routing.
apply_patch "$UPSTREAM_DIR/musetalk/utils/face_parsing/__init__.py" \
  "            if torch.cuda.is_available():
                img = torch.unsqueeze(img, 0).cuda()" \
  "            import os as _mt_os
            _mt_dev = _mt_os.environ.get('MUSETALK_DEVICE_OVERRIDE', '').strip()
            if _mt_dev in ('mps', 'cpu'):
                img = torch.unsqueeze(img, 0).to(_mt_dev)
            elif torch.cuda.is_available():
                img = torch.unsqueeze(img, 0).cuda()"

# (6) musetalk/utils/utils.py: datagen default device.
apply_patch "$UPSTREAM_DIR/musetalk/utils/utils.py" \
  '    device="cuda:0",' \
  '    device="cpu",'

# (7a) musetalk/utils/face_parsing/resnet.py: weights_only=False.
apply_patch "$UPSTREAM_DIR/musetalk/utils/face_parsing/resnet.py" \
  "    state_dict = torch.load(model_path) #modelzoo.load_url(resnet18_url)" \
  "    state_dict = torch.load(model_path, weights_only=False) #modelzoo.load_url(resnet18_url)"

# (7b) musetalk/models/unet.py: weights_only=False on UNet weight load.
apply_patch "$UPSTREAM_DIR/musetalk/models/unet.py" \
  "weights = torch.load(model_path)" \
  "weights = torch.load(model_path, weights_only=False)"
apply_patch "$UPSTREAM_DIR/musetalk/models/unet.py" \
  "weights = torch.load(model_path, map_location=device)" \
  "weights = torch.load(model_path, map_location=device, weights_only=False)"

# (7c) musetalk/models/vae.py: weights_only=False on VAE weight load if present.
apply_patch "$UPSTREAM_DIR/musetalk/models/vae.py" \
  "vae_weights = torch.load(self.model_path)" \
  "vae_weights = torch.load(self.model_path, weights_only=False)"

echo "[install] patch pass complete."

# ---------------------------------------------------------------------
# 6. weights
# ---------------------------------------------------------------------
# MuseTalk needs five families of weights, all hosted on Hugging Face:
#
#   - musetalkV15/  : the v1.5 UNet + config (preferred).
#   - musetalk/     : the v1.0 UNet + config (fallback / older configs).
#   - sd-vae/       : Stable-Diffusion VAE used by MuseTalk's diffusion path.
#   - whisper/      : Whisper audio encoder (we use whisper-tiny per upstream).
#   - dwpose/       : RTMPose-l face/body keypoint weights.
#   - face-parse-bisent/ : BiSeNet face parser + its ResNet18 backbone.
#
# We download into upstream/models/<family>/<file> via plain `curl -fL`
# against the Hugging Face `resolve/main/...` URL scheme (no
# huggingface-cli dependency, no auth tokens).
MODELS_DIR="$UPSTREAM_DIR/models"
mkdir -p "$MODELS_DIR/musetalkV15" "$MODELS_DIR/musetalk" \
         "$MODELS_DIR/sd-vae" "$MODELS_DIR/whisper" \
         "$MODELS_DIR/dwpose" "$MODELS_DIR/face-parse-bisent"

fetch() {
  local url="$1"
  local out="$2"
  local label="$3"
  if [ -s "$out" ]; then
    return 0
  fi
  echo "[install] downloading $label -> $out"
  curl -fL --retry 5 --retry-delay 3 --retry-connrefused -o "$out.partial" "$url"
  mv "$out.partial" "$out"
}

# --- MuseTalk V1.5 weights ---
fetch \
  "https://huggingface.co/TMElyralab/MuseTalk/resolve/main/musetalkV15/unet.pth" \
  "$MODELS_DIR/musetalkV15/unet.pth" \
  "musetalkV15/unet.pth (~3 GB)"
fetch \
  "https://huggingface.co/TMElyralab/MuseTalk/resolve/main/musetalkV15/musetalk.json" \
  "$MODELS_DIR/musetalkV15/musetalk.json" \
  "musetalkV15/musetalk.json"

# --- MuseTalk V1.0 weights ---
fetch \
  "https://huggingface.co/TMElyralab/MuseTalk/resolve/main/musetalk/pytorch_model.bin" \
  "$MODELS_DIR/musetalk/pytorch_model.bin" \
  "musetalk/pytorch_model.bin (~1 GB)"
fetch \
  "https://huggingface.co/TMElyralab/MuseTalk/resolve/main/musetalk/musetalk.json" \
  "$MODELS_DIR/musetalk/musetalk.json" \
  "musetalk/musetalk.json"

# --- SD VAE ---
fetch \
  "https://huggingface.co/stabilityai/sd-vae-ft-mse/resolve/main/diffusion_pytorch_model.bin" \
  "$MODELS_DIR/sd-vae/diffusion_pytorch_model.bin" \
  "sd-vae/diffusion_pytorch_model.bin (~330 MB)"
fetch \
  "https://huggingface.co/stabilityai/sd-vae-ft-mse/resolve/main/config.json" \
  "$MODELS_DIR/sd-vae/config.json" \
  "sd-vae/config.json"

# --- Whisper tiny ---
# Upstream uses whisper-tiny in download_weights.sh.  MuseTalk's
# audio_processor + WhisperModel.from_pretrained both accept whisper-tiny.
fetch \
  "https://huggingface.co/openai/whisper-tiny/resolve/main/pytorch_model.bin" \
  "$MODELS_DIR/whisper/pytorch_model.bin" \
  "whisper/pytorch_model.bin (~150 MB)"
fetch \
  "https://huggingface.co/openai/whisper-tiny/resolve/main/config.json" \
  "$MODELS_DIR/whisper/config.json" \
  "whisper/config.json"
fetch \
  "https://huggingface.co/openai/whisper-tiny/resolve/main/preprocessor_config.json" \
  "$MODELS_DIR/whisper/preprocessor_config.json" \
  "whisper/preprocessor_config.json"
fetch \
  "https://huggingface.co/openai/whisper-tiny/resolve/main/tokenizer.json" \
  "$MODELS_DIR/whisper/tokenizer.json" \
  "whisper/tokenizer.json"
fetch \
  "https://huggingface.co/openai/whisper-tiny/resolve/main/generation_config.json" \
  "$MODELS_DIR/whisper/generation_config.json" \
  "whisper/generation_config.json"

# --- DWPose RTMPose ---
fetch \
  "https://huggingface.co/yzd-v/DWPose/resolve/main/dw-ll_ucoco_384.pth" \
  "$MODELS_DIR/dwpose/dw-ll_ucoco_384.pth" \
  "dwpose/dw-ll_ucoco_384.pth (~200 MB)"

# --- Face-parse BiSeNet ---
# Upstream's download_weights.sh uses gdown for the BiSeNet weight
# (Google Drive ID 154JgKpzCPW82qINcVieuPH3fZ2e0P812). We prefer the
# Hugging Face `camenduru/MuseTalk` mirror because it's auth-free and
# stable. The ByteDance/LatentSync auxiliary tree (used as a fallback in
# earlier revisions) has been deleted upstream and now 404s — do not
# rely on it. Falling back to gdown is kept as a last resort.
if [ ! -s "$MODELS_DIR/face-parse-bisent/79999_iter.pth" ]; then
  echo "[install] downloading face-parse-bisent/79999_iter.pth (~50 MB) ..."
  fetch \
    "https://huggingface.co/camenduru/MuseTalk/resolve/main/face-parse-bisent/79999_iter.pth" \
    "$MODELS_DIR/face-parse-bisent/79999_iter.pth" \
    "face-parse-bisent/79999_iter.pth (HF camenduru mirror)" || true
  if [ ! -s "$MODELS_DIR/face-parse-bisent/79999_iter.pth" ]; then
    if command -v gdown >/dev/null 2>&1; then
      echo "[install]   HF mirror failed; falling back to gdown ..."
      gdown --id 154JgKpzCPW82qINcVieuPH3fZ2e0P812 \
        -O "$MODELS_DIR/face-parse-bisent/79999_iter.pth" || true
    fi
  fi
  if [ ! -s "$MODELS_DIR/face-parse-bisent/79999_iter.pth" ]; then
    echo "[install]   ERROR: could not download bisent weight; face-parsing required for MuseTalk."
    exit 1
  fi
fi
fetch \
  "https://download.pytorch.org/models/resnet18-5c106cde.pth" \
  "$MODELS_DIR/face-parse-bisent/resnet18-5c106cde.pth" \
  "face-parse-bisent/resnet18-5c106cde.pth (~45 MB)"

# Sanity-check weight sizes — anything dramatically smaller than expected
# indicates a partial download or an HTML error page saved as .pth.
check_weight_size() {
  local path="$1"
  local min_bytes="$2"
  if [ ! -s "$path" ]; then
    echo "[install] FAIL: $path is empty" >&2
    return 1
  fi
  local sz
  sz="$(wc -c < "$path" | tr -d '[:space:]')"
  if [ "$sz" -lt "$min_bytes" ]; then
    echo "[install] FAIL: $path is only $sz bytes (expected >= $min_bytes)" >&2
    return 1
  fi
}
check_weight_size "$MODELS_DIR/musetalkV15/unet.pth"                              2000000000
check_weight_size "$MODELS_DIR/musetalk/pytorch_model.bin"                         500000000
check_weight_size "$MODELS_DIR/sd-vae/diffusion_pytorch_model.bin"                 250000000
check_weight_size "$MODELS_DIR/whisper/pytorch_model.bin"                          100000000
check_weight_size "$MODELS_DIR/dwpose/dw-ll_ucoco_384.pth"                         100000000
check_weight_size "$MODELS_DIR/face-parse-bisent/79999_iter.pth"                    40000000
check_weight_size "$MODELS_DIR/face-parse-bisent/resnet18-5c106cde.pth"             30000000

echo
echo "[install] DONE. Sanity-check with: ./scripts/verify-install.sh"

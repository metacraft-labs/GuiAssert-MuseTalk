# MuseTalk patches for the GuiAssert-MuseTalk plugin

`python/upstream/` is a checkout of `TMElyralab/MuseTalk` pinned in
`COMMIT.txt`. MuseTalk is CUDA-first; running it on Apple-Silicon MPS
requires (1) a different dependency stack from the upstream
`requirements.txt` and (2) a handful of source patches the install
script applies after cloning. This document records the full manifest
so the install is reproducible.

## Why the heavy patching?

* **PyTorch + mmcv compatibility.** MuseTalk's upstream notes call for
  PyTorch 2.0.1, which has macOS-arm64 wheels — but its companion
  `mmcv` prebuilt wheels are CUDA-targeted and refuse to install on
  Apple Silicon. The OpenMIM ecosystem ships `mmcv-lite`, a CPU-only
  build that exposes the same Python-side APIs without the CUDA ops —
  but **mmcv-lite is not sufficient**: mmpose 1.3.2 imports
  `MultiScaleDeformableAttention` from `mmcv.ops` at module-import
  time via `mmpose.models.heads.transformer_heads.EDPoseHead`. That
  symbol lives in the compiled `mmcv._ext` extension which only the
  full `mmcv` build ships. With `mmcv-lite` installed every
  `from mmpose.apis import init_model, inference_topdown` raises
  `ModuleNotFoundError: No module named 'mmcv._ext'`.
  We therefore build full `mmcv==2.1.0` from source with
  `MMCV_WITH_OPS=1 FORCE_CUDA=0`. See "Building full mmcv on Apple
  Silicon" below for the toolchain caveats.
* **PyTorch 2.0.1 vs current stable.** torch 2.0.1 arm64 wheels are
  flaky on recent macOS; the install script defaults to
  `torch==2.2.2` (which still matches MuseTalk's checkpoint pickle
  format because they are state-dict-only) and falls back to 2.0.1
  on Linux if explicitly requested. The model weights themselves are
  unchanged.
* **TensorFlow / Gradio**. Upstream pins `tensorflow==2.12.0` for the
  training-only code path and `gradio==5.24.0` for the demo. Both are
  large and Apple-Silicon-hostile (tensorflow-arm wheels are managed
  separately by `tensorflow-macos` / `tensorflow-metal`). Neither is
  needed for inference. We drop both from `requirements-patched.txt`.
* **numpy 2.x artefacts.** The MuseTalk source still uses `np.int`-style
  aliases removed in numpy 2.x. We pin numpy 1.23.5 — matching upstream
  — and avoid that whole issue.
* **MPS device routing.** Several MuseTalk modules hardcode `cuda`
  through `torch.cuda.is_available()`. The patches below add an
  `MUSETALK_DEVICE_OVERRIDE` env hook so the wrapper script can force
  `mps` (or `cpu`) without modifying upstream logic.
* **`PYTORCH_ENABLE_MPS_FALLBACK=1`** must be set when running on MPS:
  a handful of `aten::*` kernels invoked by the diffusers VAE + mmpose
  RTMPose backbone are not yet implemented natively for MPS and need
  to silently fall back to CPU. The wrapper sets this automatically.

## requirements-patched.txt

See the sibling file. It keeps the inference-relevant pins (`diffusers`,
`accelerate`, `transformers`, `huggingface_hub`, `librosa`, `soundfile`,
`numpy==1.23.5`) at upstream values, drops `tensorflow`, `tensorboard`,
`gradio`, and leaves `scipy` open. The `mmengine` / `mmdet` / `mmpose`
trio is installed via OpenMIM (`mim install`) in step 3 of
`scripts/install.sh`, NOT via this file, because pip's dependency
solver cannot reliably resolve them on Apple Silicon. Full `mmcv`
itself is built from source in the same step (see "Building full mmcv
on Apple Silicon" below) — `mmcv-lite` is not sufficient because
`mmpose 1.3.2` needs `mmcv._ext`'s `MultiScaleDeformableAttention`.

## Install-time environment hazards

### PYTHONPATH leakage from a Nix/direnv shell

Several Apple-Silicon developer setups (notably Nix + direnv layered
shells) export `PYTHONPATH` pointing at a Nix-store Python 3.13
site-packages directory. That path shadows the venv's Python 3.10 stdlib
— most importantly the `sysconfig` module. The leaked `sysconfig`
returns `VERSION='3.13'` and `SOABI='cpython-313-darwin'`, which makes
every C-extension wheel build fail with
`AssertionError: would build wheel with unsupported tag ('cp310', 'cp313', 'macosx_*_arm64')`.

`scripts/install.sh` unsets `PYTHONPATH` and `PYTHONHOME` at startup so
the venv's Python 3.10 stdlib is the only one on the import path. If
you ever invoke `pip install ...` against the venv outside the
install script, do the same:

    unset PYTHONPATH PYTHONHOME
    source .venv/bin/activate
    pip install ...

### `mmcv` upper bound (< 2.2.0)

mmdet 3.x's `__init__.py` carries an explicit assertion:

    assert (digit_version(mmcv.__version__) >= digit_version(mmcv_minimum_version)
            and mmcv.__version__ < digit_version(mmcv_maximum_version)), \
           'MMCV==X.Y.Z is used but incompatible. Please install mmcv>=2.0.0rc4, <2.2.0.'

If `mmcv` resolves to 2.2.0 (the latest as of writing), every
`import mmdet` aborts with that assertion. The install script pins
`mmcv==2.1.0` and opportunistically uninstalls a too-new pre-existing
copy before reinstalling.

### Building full mmcv on Apple Silicon

There are no pre-built wheels for full `mmcv` 2.x on macOS-arm64; pip
must compile the `mmcv/_ext.*.so` C++ extension from source. Two
toolchain hazards have to be worked around:

1. **`--no-build-isolation` is mandatory.** mmcv's `setup.py` invokes
   the venv's installed PyTorch to discover its C++ headers. The
   isolated build overlay hides those — leaving the build with no
   `torch/extension.h`. Pass `--no-build-isolation` (and pre-install
   the build prerequisites — pytorch + cython + numpy + setuptools —
   into the venv first).

2. **libc++ `_LIBCPP_NO_SPECIALIZATIONS` in the macOS 26 SDK.** Apple's
   clang 21 / macOS 26 SDK marks several standard traits
   (`is_arithmetic`, ...) as `_LIBCPP_NO_SPECIALIZATIONS`, which
   makes PyTorch 2.2's `c10::util::strong_type` specialisations a hard
   build error:

       error: 'is_arithmetic' cannot be specialized: Users are not
       allowed to specialize this standard library entity

   We pin `-isysroot` to the older `MacOSX15.sdk` (whose libc++
   headers don't carry the no-specializations annotation) via
   `CFLAGS`/`CXXFLAGS`/`LDFLAGS` and `MACOSX_DEPLOYMENT_TARGET=14.0`.
   The Apple-shipped clang under
   `/Library/Developer/CommandLineTools/usr/bin/clang` (with
   `-isysroot .../MacOSX15.sdk`) builds mmcv 2.1.0 in ~3-6 min with
   `MAX_JOBS=2`.

   We deliberately avoid the Nix-store clang (when running under
   `nix develop`): its bundled libcxx triggers a separate
   `unresolved using declaration` failure in `<chrono>`/`<ratio>` for
   the same PyTorch headers. The install script always uses the Apple
   clang explicitly.

   If `MacOSX15.sdk` isn't installed (only `MacOSX26.sdk` is
   available), the build will fail. Install Xcode Command Line Tools
   that include MacOSX15.sdk.

### `chumpy` build with `--no-build-isolation`

`chumpy` is an mmpose transitive dep (used by SMPL utilities). Its
`setup.py` does `import pip` directly inside the build process. Modern
pip's isolated build environment does not expose `pip` to setup.py and
the build aborts with `ModuleNotFoundError: No module named 'pip'`.
The install script pre-installs chumpy with `--no-build-isolation` so
its setup.py runs against the venv's pip.

### `xtcocotools` build with `--no-build-isolation`

`xtcocotools` (an mmpose transitive dep — Extended COCO API for keypoint
metrics) has `import numpy` at the top of its `setup.py` and a Cython
extension module that ships only `.pyx` source (not pre-Cythonised
`.c`). The isolated build env lacks numpy and Cython, so the build
fails. The install script pre-installs `cython>=0.27.3` then installs
xtcocotools with `--no-build-isolation` so setup.py sees the venv's
numpy + Cython.

### Soft mmpose ↔ mmdet version warning

`mmpose 1.3.2`'s metadata declares `mmdet < 3.3.0` in its `mim` extra,
but the install script pins `mmdet >= 3.0.0, < 3.4.0` and the resolver
lands on `mmdet 3.3.0`. pip prints a non-fatal dependency-conflict
warning. mmpose's runtime use of mmdet (RTMPose face detection) works
correctly with mmdet 3.3.0 in practice; the warning is suppressed-not-
fixed by upstream and does not block inference.

## Source patches applied to `upstream/`

The install script applies each patch idempotently via a small Python
helper that does literal-string replacement and skips already-rewritten
files. The patches are small and the legacy needles are unambiguous.

### 1. `scripts/inference.py` — device selection honours `MUSETALK_DEVICE_OVERRIDE`

Upstream picks `cuda:<gpu_id>` if CUDA is available, else `cpu`. We
extend that to honour an environment variable so the wrapper can force
`mps` (or `cpu`) on Apple Silicon:

    -    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    +    import os as _mt_os
    +    _mt_dev = _mt_os.environ.get('MUSETALK_DEVICE_OVERRIDE', '').strip()
    +    if _mt_dev in ('mps', 'cpu'):
    +        device = torch.device(_mt_dev)
    +    elif _mt_dev.startswith('cuda'):
    +        device = torch.device(_mt_dev)
    +    else:
    +        device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")

### 2. `musetalk/utils/preprocessing.py` — mmpose + face-alignment device routing

Two top-level device assignments hardcode `cuda`. We patch both:

    -device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    +import os as _mt_os
    +_mt_dev = _mt_os.environ.get('MUSETALK_DEVICE_OVERRIDE', '').strip()
    +if _mt_dev in ('mps', 'cpu') or _mt_dev.startswith('cuda'):
    +    device = torch.device(_mt_dev)
    +else:
    +    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

(applied to the first occurrence — the mmpose `init_model` device)

    -device = "cuda" if torch.cuda.is_available() else "cpu"
    +device = os.environ.get('MUSETALK_DEVICE_OVERRIDE', '').strip() or \
    +    ("cuda" if torch.cuda.is_available() else "cpu")

(applied to the second occurrence — the FaceAlignment device string)

### 3. `musetalk/utils/face_detection/detection/core.py` — accept `mps` device

`FaceDetector.__init__` allow-lists `cpu` and `cuda` and raises
`ValueError` otherwise. The s3fd convolutions run fine on MPS under
`PYTORCH_ENABLE_MPS_FALLBACK=1`, so we extend the allow-list:

    -        if 'cpu' not in device and 'cuda' not in device:
    +        if 'cpu' not in device and 'cuda' not in device and 'mps' not in device:

### 4. `musetalk/utils/face_detection/api.py` — accept `mps` in FaceAlignment

The same pattern lives one layer up in `FaceAlignment.__init__`:

    -        if 'cuda' in device:
    +        if 'cuda' in device or 'mps' in device:

(`'mps' in device` triggers the same `torch.backends.*` configuration
branch as `'cuda' in device` — both want non-CPU code paths, and the
fallback env covers any unsupported kernel.)

### 5. `musetalk/utils/face_parsing/__init__.py` — MPS-aware face-parser load

The `FaceParsing.__init__` hardcodes `cuda` for both the weight load
and the per-image inference call:

    -        if torch.cuda.is_available():
    -            net.cuda()
    -            net.load_state_dict(torch.load(model_pth))
    -        else:
    -            net.load_state_dict(torch.load(model_pth, map_location=torch.device('cpu')))
    +        import os as _mt_os
    +        _mt_dev = _mt_os.environ.get('MUSETALK_DEVICE_OVERRIDE', '').strip()
    +        if _mt_dev in ('mps', 'cpu'):
    +            net.to(_mt_dev)
    +            net.load_state_dict(torch.load(model_pth, map_location=torch.device(_mt_dev), weights_only=False))
    +        elif torch.cuda.is_available():
    +            net.cuda()
    +            net.load_state_dict(torch.load(model_pth, weights_only=False))
    +        else:
    +            net.load_state_dict(torch.load(model_pth, map_location=torch.device('cpu'), weights_only=False))

(also flipping `weights_only=False` to survive the PyTorch 2.6 default
flip on the pre-2.6 state-dict pickle.)

**Whitespace gotcha.** Upstream ships this block with a literal
trailing space after `net.load_state_dict(torch.load(model_pth)) `
(line 65 in the pinned revision). Earlier revisions of
`scripts/install.sh` used a strict `grep -qF` literal match — the
trailing space made the needle miss and the patch silently no-oped,
which then surfaced as a `FileNotFoundError` on
`./models/face-parse-bisent/79999_iter.pth` on Apple Silicon (the
unpatched branch always took the CUDA path and stayed inside the
broken `models/...` literal). The current `apply_patch` helper builds
a regex that allows arbitrary trailing whitespace at the end of every
needle line, so the patch fires on both the upstream layout and any
locally-cleaned variant.

The per-image path:

    -            if torch.cuda.is_available():
    -                img = torch.unsqueeze(img, 0).cuda()
    +            import os as _mt_os
    +            _mt_dev = _mt_os.environ.get('MUSETALK_DEVICE_OVERRIDE', '').strip()
    +            if _mt_dev in ('mps', 'cpu'):
    +                img = torch.unsqueeze(img, 0).to(_mt_dev)
    +            elif torch.cuda.is_available():
    +                img = torch.unsqueeze(img, 0).cuda()

### 6. `musetalk/utils/utils.py` — datagen default device

Trivial: change the default kwarg from `cuda:0` to `cpu` so an
explicit `device=...` arg is mandatory. The caller in `scripts/inference.py`
already passes `device=device`, so this only affects users that import
`datagen` directly:

    -    device="cuda:0",
    +    device="cpu",

### 7. `torch.load` `weights_only` flips

PyTorch 2.6 flipped the `torch.load(..., weights_only=...)` default
from `False` to `True` to harden against arbitrary-code execution from
malicious pickle files. Several MuseTalk modules load pre-2.6
state-dict pickles via the legacy API. We flip the default back to
`False` at every load site we touch:

* `musetalk/utils/face_parsing/__init__.py` (covered by patch 5).
* `musetalk/utils/face_parsing/resnet.py`:

      -        state_dict = torch.load(model_path)
      +        state_dict = torch.load(model_path, weights_only=False)

* `musetalk/models/unet.py` (UNet weight load — pinned by inspecting
  the file at the pinned SHA; the install script's `apply_patch`
  helper skips this transparently when the needle is absent on a
  future upstream revision):

      -            state_dict = torch.load(self.model_path)
      +            state_dict = torch.load(self.model_path, weights_only=False)

### 8. `whisper-small` weight pre-pull

MuseTalk uses `transformers.WhisperModel.from_pretrained(args.whisper_dir)`.
When `args.whisper_dir` points at a local directory (which our wrapper
ensures), `transformers` does not network-fetch. The install script
pre-downloads `openai/whisper-small` (or `whisper-tiny` per upstream's
`download_weights.sh`) into `models/whisper/`. At inference time we set
`HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1` to be defensive.

### 9. Inference config

MuseTalk's `scripts/inference.py` reads an OmegaConf YAML at
`--inference_config` and iterates over its top-level keys as task IDs.
Each task entry must contain `video_path` and `audio_path`. The wrapper
generates a one-shot config on the fly that points at the caller's
portrait + narration:

    task_0:
      video_path: "<caller-supplied portrait or video>"
      audio_path: "<caller-supplied wav>"

and passes `--result_dir <out-dir>` + `--output_vid_name <name>.mp4`
so the produced file lands exactly where the caller asked.

## Re-applying the patches

If `python/upstream/` is wiped and re-cloned, the patches above must be
reapplied. `scripts/install.sh` does this automatically (and is
idempotent — re-running it skips already-patched files because each
`apply_patch` invocation greps for the legacy needle and skips when it's
missing).

## Apple Silicon performance notes

With these patches and `PYTORCH_ENABLE_MPS_FALLBACK=1`, MuseTalk on
MPS renders at roughly 3-10x slower than the CUDA path quoted by
upstream. Realistic numbers on an M-series Mac:

  * Model load + face-detect warmup: ~30-60 s wall-clock.
  * Per-frame inference: ~0.5-2 s/frame (vs ~0.1 s/frame on RTX 4090).
  * 5 s of narration -> 125 frames at 25 FPS -> ~2-4 min total.

This is workable for the test fixture and for off-line production
renders. Real-time use on MPS is not viable.

## Weights

The install script downloads (~5 GB total):

  * `models/musetalkV15/unet.pth` + `musetalk.json` from
    `huggingface.co/TMElyralab/MuseTalk` (~3 GB).
  * `models/musetalk/pytorch_model.bin` + `musetalk.json` (v1.0) from
    the same repo (~1 GB).
  * `models/sd-vae/diffusion_pytorch_model.bin` + `config.json` from
    `huggingface.co/stabilityai/sd-vae-ft-mse` (~330 MB).
  * `models/whisper/pytorch_model.bin` + `config.json` +
    `preprocessor_config.json` from `huggingface.co/openai/whisper-tiny`
    (~150 MB; upstream's `download_weights.sh` uses `whisper-tiny`).
  * `models/dwpose/dw-ll_ucoco_384.pth` from
    `huggingface.co/yzd-v/DWPose` (~200 MB).
  * `models/face-parse-bisent/79999_iter.pth` (~50 MB) +
    `models/face-parse-bisent/resnet18-5c106cde.pth` (~45 MB).

Sources are documented inline in `scripts/install.sh`. The musetalk /
sd-vae / whisper / dwpose weights come from Hugging Face mirrors of
the corresponding upstream repos (`TMElyralab/MuseTalk`,
`stabilityai/sd-vae-ft-mse`, `openai/whisper-tiny`, `yzd-v/DWPose`).
The BiSeNet face-parser weight (`79999_iter.pth`) originates from a
Google Drive ID in upstream's `download_weights.sh`. We mirror it via
the `camenduru/MuseTalk` Hugging Face repo (auth-free); the
`ByteDance/LatentSync` auxiliary tree that earlier revisions used as a
fallback has since been deleted upstream and returns 404. The
ResNet18 backbone weight is pulled directly from
`download.pytorch.org`. No `gdown` dependency is required in the
common path; `gdown` is only consulted as a last-resort fallback if
both Hugging Face mirrors are unreachable.

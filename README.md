# GuiAssert-MuseTalk

MuseTalk talking-head plugin for [GuiAssert]. Implements GuiAssert's
`TalkingHeadProvider` contract by shelling out to a local MuseTalk
install (a Python 3.10 venv + the `TMElyralab/MuseTalk` repo at a
pinned commit + ~5 GB of model weights from Hugging Face).

MuseTalk is a 2024 latent-diffusion talking-head model from Tencent's
MuseV team ("Real-Time High Quality Lip Synchronization with Latent
Space Inpainting"). On CUDA it can drive a 30 FPS lip-sync stream off a
still portrait with notably higher per-frame quality than older
models like Wav2Lip. The trade-off is dependency weight: it relies on
the OpenMIM family (`mmpose` for face landmarks, full `mmcv` for the
underlying ops, `mmdet` for detection), the Stable-Diffusion VAE for
latent decoding, and the OpenAI Whisper encoder for audio embeddings.
On Apple Silicon there are no pre-built `mmcv` wheels, so the install
script compiles `mmcv==2.1.0` from source — see `python/PATCHES.md`
for the toolchain workarounds.

This repository is intentionally heavyweight. By keeping it separate
from GuiAssert, any caller that only wants the lightweight
`stock_avatar` placeholder avoids paying the MuseTalk dependency cost.
For a smaller, faster, less-quality alternative see the sibling
`GuiAssert-Wav2Lip` plugin; for head-motion in addition to lip-sync see
`GuiAssert-SadTalker`.

[GuiAssert]: ../GuiAssert/

## Layout

```
GuiAssert-MuseTalk/
├── flake.nix                          python3 + nim + git + curl + ffmpeg-full + cmake + pkg-config devShell
├── gui_assert_musetalk.nimble         nimble package
├── src/
│   └── gui_assert_musetalk.nim        plugin implementation (TalkingHeadProvider)
├── python/
│   ├── render_musetalk.py             MuseTalk CLI wrapper
│   ├── requirements-patched.txt       deps tuned for Python 3.10 + PyTorch 2.x on Apple Silicon
│   ├── PATCHES.md                     patch manifest applied to the upstream checkout
│   ├── COMMIT.txt                     pinned upstream SHA
│   └── upstream/                      (gitignored) MuseTalk clone — populated by install.sh
├── scripts/
│   ├── install.sh                     create .venv, clone upstream, apply patches, fetch weights
│   └── verify-install.sh              smoke-test the install
└── tests/
    ├── fixtures/
    │   ├── README.md                  provenance
    │   └── portrait.png               PD-US-expired Einstein portrait, 400x400
    └── tmusetalk.nim                  pure tests + `-d:musetalkLive` gated live test
```

## Cost of setup

| Resource     | Approx.                                                |
| ------------ | ------------------------------------------------------ |
| Disk         | ~5 GB of weights + ~3 GB of Python deps in `.venv`     |
| Network      | ~5 GB on first install (subsequent runs are offline)   |
| Time         | ~10-30 min on a fresh checkout (weight downloads dominate) |
| Dollars      | Zero — MuseTalk is open source (MIT-licensed code; weights on HF) |
| API key      | None                                                   |

## Setup

```sh
nix develop                  # python3 + nim + ffmpeg-full + cmake + pkg-config + git + curl
./scripts/install.sh         # ~10-30 min on a fresh checkout (weight downloads dominate)
./scripts/verify-install.sh  # quick smoke-test
```

The install script is idempotent. Re-running it skips already-done
steps and re-applies patches incrementally.

### Python 3.10 requirement

`scripts/install.sh` requires `python3.10` on `PATH`. The dev-shell's
`python3` is currently used only for utility scripts; the venv that
hosts MuseTalk's deps is created from your host `python3.10`. On macOS:

```sh
brew install python@3.10
# (the install script picks up python3.10 from /opt/homebrew/bin)
```

On Linux, install your distribution's `python3.10` package
(`apt install python3.10` on Debian/Ubuntu, etc.).

### OpenMIM dependency family

MuseTalk's preprocessing pipeline calls `mmpose.apis.init_model` +
`inference_topdown` for face landmarks. The `mmpose` family
(`mmengine`, `mmcv`, `mmdet`, `mmpose`) interlocks through tight
version constraints that pip's solver cannot navigate reliably on
Apple Silicon, so the install script uses OpenMIM (`pip install -U
openmim && mim install ...`) for `mmengine` / `mmdet` / `mmpose`.

`mmcv` itself has to be the full build (not `mmcv-lite`): `mmpose`
1.3.2 imports `MultiScaleDeformableAttention` from `mmcv.ops` at
module-import time via `EDPoseHead`, and that symbol only exists in
the compiled `mmcv._ext` extension which `mmcv-lite` does not ship.
The prebuilt `mmcv` wheels on PyPI are CUDA-targeted and won't install
on macOS arm64, so the install script compiles `mmcv==2.1.0` from
source with `MMCV_WITH_OPS=1 FORCE_CUDA=0` (~3-6 min on an
M-series Mac). See `python/PATCHES.md` ("Building full mmcv on Apple
Silicon") for the SDK / toolchain caveats.

### Apple Silicon notes

* MuseTalk runs on the MPS backend with
  `PYTORCH_ENABLE_MPS_FALLBACK=1` (set automatically by the dev-shell
  and the wrapper script). A number of `aten::*` ops fall back to CPU;
  the diffusers VAE + UNet bulk runs on the GPU.
* Typical render speed on an M-series Mac: roughly 3-10x slower than
  the CUDA path quoted by upstream. A 5-second narration (125 frames
  at 25 FPS) takes ~2-4 minutes wall-clock; first-run is slower
  because the mmpose / face-detect / VAE backbones all warm up
  individually.
* Upstream MuseTalk was authored against PyTorch 2.0.1 + CUDA. We patch
  device-selection sites to honour an `MUSETALK_DEVICE_OVERRIDE` env
  var so the wrapper can force `mps` (or `cpu`) without modifying
  upstream logic. See `python/PATCHES.md` for the full manifest.
* PyTorch 2.6+ flipped the default of `torch.load(weights_only=...)`
  to `True`, which breaks the pre-2.6 state-dict pickles MuseTalk and
  its auxiliary face-parsing / VAE weights ship. The patch manifest
  flips it back at every load site we touch.

## Wiring into a runner

```nim
import gui_assert/talking_head
import gui_assert_musetalk

let reg = newRegistry()         # registry pre-populated with `stock_avatar`
registerMuseTalk(reg)           # now `musetalk` is also registered

var opts = TalkingHeadOpts(
  avatarImagePath: some(avatarPng),
  device: "mps",
  cacheDir: some("/tmp/musetalk-cache"),
)
generateTalkingHead(reg, "musetalk", narrationWav, outputMp4, opts)
```

Path discovery uses three environment variables (each with a sensible
default):

| Variable | Default | Purpose |
| --- | --- | --- |
| `GUI_ASSERT_MUSETALK_HOME` | this repo's root | Override the plugin install location. |
| `GUI_ASSERT_MUSETALK_PYTHON` | `<home>/.venv/bin/python` | Override the Python interpreter. |
| `GUI_ASSERT_MUSETALK_RENDER_SCRIPT` | `<home>/python/render_musetalk.py` | Override the wrapper script. |

## Tests

```sh
# Pure tests — always safe to run.
nim c -r --hints:off --path:src --path:../GuiAssert/src tests/tmusetalk.nim

# Live end-to-end — requires the install to have completed.
nim c -d:musetalkLive -r --hints:off --path:src --path:../GuiAssert/src tests/tmusetalk.nim
```

The live test fails the run if MuseTalk is not actually available —
it is not a graceful skip. CI that does not want to install MuseTalk
simply compiles without `-d:musetalkLive`.

The live test synthesises its own narration WAV (via macOS `say` +
ffmpeg resample to 16 kHz mono) and reads a small portrait fixture
from `tests/fixtures/portrait.png` (override either via
`$GUI_ASSERT_MUSETALK_TEST_WAV` / `$GUI_ASSERT_MUSETALK_TEST_AVATAR`).

## License

MIT — see `LICENSE`. Upstream MuseTalk is also MIT-licensed; this
plugin does not redistribute it (the install script clones it directly
from GitHub).

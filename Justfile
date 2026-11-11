# GuiAssert-MuseTalk
#
# `just test`              - run the pure tests (no MuseTalk install needed).
# `just test-live`         - run the gated live test (-d:musetalkLive).
# `just install`           - install / refresh the local MuseTalk setup.
# `just verify`            - smoke-test the install.
# `just lint`              - placeholder; required by the workspace pre-commit hook.

# Sibling sources: DemoVideoStudio (the plugin contract) and GuiAssert (what
# the studio itself builds on), next to this checkout unless overridden.
studio_src := env_var_or_default("DEMO_VIDEO_STUDIO_SRC", "../DemoVideoStudio/src")
gui_assert_src := env_var_or_default("GUI_ASSERT_SRC", "../GuiAssert/src")

default: test

# Pure unit tests for the plugin.  Compiles against the sibling
# GuiAssert checkout via --path:{{studio_src}} --path:{{gui_assert_src}}.
test:
    nim c -r --hints:off --path:src tests/tnimcache_is_worktree_local.nim
    nim c -r --hints:off --path:src --path:{{studio_src}} --path:{{gui_assert_src}} tests/tmusetalk.nim
    nim c -r --hints:off --path:src --path:{{studio_src}} --path:{{gui_assert_src}} tests/tcontract.nim

# End-to-end live test.  Requires a completed install (.venv + upstream
# checkout + model weights) under this repo.
test-live:
    nim c -d:musetalkLive -r --hints:off --path:src --path:{{studio_src}} --path:{{gui_assert_src}} tests/tmusetalk.nim

# Install / refresh the local MuseTalk setup.
install:
    ./scripts/install.sh

# Smoke-test the install.
verify:
    ./scripts/verify-install.sh

# Required by the workspace's pre-commit hook (`just lint`).  Add real
# linters here as they come online (e.g. `nim check`).
lint:
    @echo "[lint] no linters configured yet for GuiAssert-MuseTalk."

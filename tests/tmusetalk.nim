## Unit + integration tests for the MuseTalk GuiAssert plugin.
##
## ## Pure tests (always run)
##
##   * `musetalkProvider()` produces a value with the correct name and
##     non-nil callbacks,
##   * `isAvailable()` returns false when the python binary or the
##     wrapper script are missing,
##   * `isAvailable()` returns true when both paths exist (we forge
##     dummies under tmp + point the env vars at them),
##   * cache-key determinism: same inputs -> same key; different
##     avatar / different audio / different device -> different keys,
##   * `registerMuseTalk` integrates with the registry (`hasProvider`
##     reports membership, `getProvider` returns the registered
##     value),
##   * the `generate` proc rejects requests with no `avatarImagePath`
##     before spawning Python,
##   * the `generate` proc surfaces a clear error when the python
##     binary is missing.
##
## ## Live test (compile-time-gated)
##
## When compiled with `-d:musetalkLive` we end-to-end render a real
## talking-head MP4:
##
##   nim c -d:musetalkLive -r --hints:off --path:src --path:../GuiAssert/src \
##       tests/tmusetalk.nim
##
## Requirements:
##   * a working install under `.venv/` + `python/upstream/` (see
##     `scripts/install.sh`),
##   * a narration WAV — generated on-demand via the macOS `say`
##     binary at `/tmp/musetalk-test-narration.wav`. Override via
##     `$GUI_ASSERT_MUSETALK_TEST_WAV`.
##   * a portrait fixture — `tests/fixtures/portrait.png`. Override
##     via `$GUI_ASSERT_MUSETALK_TEST_AVATAR`.
##
## The live suite never silently skips: a missing prerequisite is a
## test failure (per the project's "no graceful skips" policy).  CI
## that does not have MuseTalk installed simply does not pass
## `-d:musetalkLive`.

import std/[options, os, unittest]

import gui_assert/talking_head
import gui_assert_musetalk

when defined(musetalkLive):
  import std/[json, osproc, streams, strformat, strutils, times]

# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

proc thisRepoRoot(): string =
  ## `currentSourcePath` -> .../GuiAssert-MuseTalk/tests/tmusetalk.nim
  currentSourcePath().parentDir().parentDir()

# ---------------------------------------------------------------------------
# Pure tests
# ---------------------------------------------------------------------------

suite "musetalk provider value":

  test "musetalkProvider builds a provider with the canonical name":
    let p = musetalkProvider()
    check p.name == ProviderName
    check p.name == "musetalk"
    check (not p.isAvailable.isNil)
    check (not p.generate.isNil)

  test "registerMuseTalk exposes the plugin via the registry":
    let r = newRegistry()
    check (not hasProvider(r, "musetalk"))
    registerMuseTalk(r)
    check hasProvider(r, "musetalk")
    let got = getProvider(r, "musetalk")
    check got.name == "musetalk"
    # Stock avatar must still be present (we layer plugins; we do not
    # replace the default registry).
    check hasProvider(r, "stock_avatar")

suite "musetalk isAvailable":

  test "returns false when python / script paths are absent":
    let tmp = getTempDir() / "tmusetalk_avail_missing"
    if dirExists(tmp): removeDir(tmp)
    createDir(tmp)
    # Point env vars at non-existent paths and confirm.
    putEnv(PythonOverrideEnvVar, tmp / "nope-python")
    putEnv(ScriptOverrideEnvVar, tmp / "nope-script.py")
    check (not musetalkIsAvailable())
    delEnv(PythonOverrideEnvVar)
    delEnv(ScriptOverrideEnvVar)

  test "returns true when both paths exist":
    let tmp = getTempDir() / "tmusetalk_avail_present"
    if dirExists(tmp): removeDir(tmp)
    createDir(tmp)
    let py = tmp / "python"
    let scr = tmp / "render.py"
    writeFile(py, "#!/bin/sh\nexit 0\n")
    writeFile(scr, "print('hello')\n")
    putEnv(PythonOverrideEnvVar, py)
    putEnv(ScriptOverrideEnvVar, scr)
    check musetalkIsAvailable()
    delEnv(PythonOverrideEnvVar)
    delEnv(ScriptOverrideEnvVar)

suite "musetalk cache key":

  setup:
    let cacheTmp = getTempDir() / "tmusetalk_cachekey"
    if dirExists(cacheTmp): removeDir(cacheTmp)
    createDir(cacheTmp)
    # Two distinct avatars + two distinct audio fixtures (real bytes
    # so the SHA-1 of each input differs).
    let avatar1 = cacheTmp / "a1.png"
    let avatar2 = cacheTmp / "a2.png"
    let nar1 = cacheTmp / "n1.wav"
    let nar2 = cacheTmp / "n2.wav"
    writeFile(avatar1, "PNG-bytes-A")
    writeFile(avatar2, "PNG-bytes-B")
    writeFile(nar1, "RIFF-A")
    writeFile(nar2, "RIFF-B")

  test "same inputs produce the same key":
    let k1 = cacheKeyFor(avatar1, nar1, ProviderName, "mps")
    let k2 = cacheKeyFor(avatar1, nar1, ProviderName, "mps")
    check k1 == k2
    check k1.len == 16

  test "different avatar -> different key":
    let k1 = cacheKeyFor(avatar1, nar1, ProviderName, "mps")
    let k2 = cacheKeyFor(avatar2, nar1, ProviderName, "mps")
    check k1 != k2

  test "different audio -> different key":
    let k1 = cacheKeyFor(avatar1, nar1, ProviderName, "mps")
    let k2 = cacheKeyFor(avatar1, nar2, ProviderName, "mps")
    check k1 != k2

  test "different device -> different key":
    let k1 = cacheKeyFor(avatar1, nar1, ProviderName, "mps")
    let k2 = cacheKeyFor(avatar1, nar1, ProviderName, "cpu")
    check k1 != k2

suite "musetalk generate input validation":

  test "missing avatarImagePath raises TalkingHeadError":
    # Pretend the install is present so we reach the avatar check.
    let tmp = getTempDir() / "tmusetalk_no_avatar"
    if dirExists(tmp): removeDir(tmp)
    createDir(tmp)
    let py = tmp / "python"
    let scr = tmp / "render.py"
    writeFile(py, "#!/bin/sh\nexit 0\n")
    writeFile(scr, "print('hi')\n")
    putEnv(PythonOverrideEnvVar, py)
    putEnv(ScriptOverrideEnvVar, scr)
    try:
      let r = newRegistry()
      registerMuseTalk(r)
      let nar = tmp / "n.wav"
      writeFile(nar, "RIFF")
      let outMp4 = tmp / "out.mp4"
      let opts = TalkingHeadOpts(avatarImagePath: none(string),
                                 cacheDir: some(tmp / "cache"))
      expect TalkingHeadError:
        generateTalkingHead(r, "musetalk", nar, outMp4, opts)
    finally:
      delEnv(PythonOverrideEnvVar)
      delEnv(ScriptOverrideEnvVar)

  test "missing python binary surfaces a clear error":
    # Pretend the install is missing (env vars unset, pluginRoot's
    # default path won't have a .venv in a freshly-cloned tree
    # without the install script having been run).
    let tmp = getTempDir() / "tmusetalk_no_install"
    if dirExists(tmp): removeDir(tmp)
    createDir(tmp)
    # Force pluginRoot to point somewhere with no .venv.
    putEnv(PluginRootEnvVar, tmp)
    try:
      let r = newRegistry()
      registerMuseTalk(r)
      # Registering the provider should not throw, but dispatch
      # should fail availability first.
      check (not musetalkIsAvailable())
      let avatar = tmp / "a.png"
      writeFile(avatar, "PNG-bytes")
      let nar = tmp / "n.wav"
      writeFile(nar, "RIFF")
      let outMp4 = tmp / "out.mp4"
      let opts = TalkingHeadOpts(avatarImagePath: some(avatar),
                                 cacheDir: some(tmp / "cache"))
      expect TalkingHeadError:
        generateTalkingHead(r, "musetalk", nar, outMp4, opts)
    finally:
      delEnv(PluginRootEnvVar)

# ---------------------------------------------------------------------------
# Live test — compile-time-gated.
# ---------------------------------------------------------------------------
when defined(musetalkLive):

  proc ffprobeJson(path: string): JsonNode =
    let ffprobe =
      block:
        let env = getEnv("FFPROBE_BIN")
        if env.len > 0 and fileExists(env): env
        else: findExe("ffprobe")
    doAssert ffprobe.len > 0 and fileExists(ffprobe),
      "ffprobe not on PATH; install ffmpeg to run the live MuseTalk test."
    let p = startProcess(
      command = ffprobe,
      args = @["-hide_banner", "-v", "error", "-print_format", "json",
               "-show_streams", "-show_format", path],
      options = {poStdErrToStdOut}
    )
    let raw = p.outputStream.readAll()
    let code = p.waitForExit()
    p.close()
    doAssert code == 0, "ffprobe failed (" & $code & "): " & raw
    parseJson(raw)

  proc runSh(args: openArray[string]): tuple[code: int, output: string] =
    ## Run a process and capture combined stdout+stderr.
    let bin = findExe(args[0])
    doAssert bin.len > 0, "binary not on PATH: " & args[0]
    var rest: seq[string] = @[]
    for i in 1 ..< args.len: rest.add args[i]
    let p = startProcess(
      command = bin, args = rest, options = {poStdErrToStdOut}
    )
    let raw = p.outputStream.readAll()
    let code = p.waitForExit()
    p.close()
    result = (code: code, output: raw)

  proc ensureNarrationWav(): string =
    ## Generate a short WAV via macOS `say` if one isn't supplied.
    let envOverride = getEnv("GUI_ASSERT_MUSETALK_TEST_WAV")
    if envOverride.len > 0:
      doAssert fileExists(envOverride),
        "GUI_ASSERT_MUSETALK_TEST_WAV points at a non-existent path: " &
        envOverride
      return envOverride
    let target = "/tmp/musetalk-test-narration.wav"
    if fileExists(target) and getFileSize(target) > 1024:
      return target
    # macOS `say` produces AIFF by default; pipe through ffmpeg into
    # a 16 kHz mono WAV (MuseTalk's whisper-tiny encoder demands 16 kHz).
    let aiff = "/tmp/musetalk-test-narration.aiff"
    let sayBin = findExe("say")
    doAssert sayBin.len > 0,
      "macOS `say` binary missing; set GUI_ASSERT_MUSETALK_TEST_WAV to a WAV path."
    let phrase = "Hello from GuiAssert MuseTalk. This is a test render."
    let r1 = runSh([sayBin, "-o", aiff, phrase])
    doAssert r1.code == 0, "say failed: " & r1.output
    let ffBin =
      block:
        let env = getEnv("FFMPEG_BIN")
        if env.len > 0 and fileExists(env): env
        else: findExe("ffmpeg")
    doAssert ffBin.len > 0, "ffmpeg missing"
    if fileExists(target): removeFile(target)
    let r2 = runSh([ffBin, "-hide_banner", "-loglevel", "error", "-y",
                    "-i", aiff, "-ar", "16000", "-ac", "1", target])
    doAssert r2.code == 0, "ffmpeg resample failed: " & r2.output
    result = target

  proc ensurePortrait(): string =
    ## Use the bundled portrait fixture unless the caller overrides it.
    let envOverride = getEnv("GUI_ASSERT_MUSETALK_TEST_AVATAR")
    if envOverride.len > 0:
      doAssert fileExists(envOverride),
        "GUI_ASSERT_MUSETALK_TEST_AVATAR points at a non-existent path: " &
        envOverride
      return envOverride
    let bundled = thisRepoRoot() / "tests" / "fixtures" / "portrait.png"
    doAssert fileExists(bundled),
      "no portrait fixture at " & bundled &
      " (set GUI_ASSERT_MUSETALK_TEST_AVATAR to override)"
    result = bundled

  suite "musetalk live talking-head render":

    test "renders a real talking-head MP4 and round-trips the cache":
      let avatar = ensurePortrait()
      let narration = ensureNarrationWav()

      doAssert musetalkIsAvailable(),
        "MuseTalk not available — run scripts/install.sh first."

      let tmp = getTempDir() / "tmusetalk_live"
      if dirExists(tmp): removeDir(tmp)
      createDir(tmp)

      let r = newRegistry()
      registerMuseTalk(r)

      let outMp4 = tmp / "live.mp4"
      let opts = TalkingHeadOpts(
        avatarImagePath: some(avatar),
        device: "mps",
        cacheDir: some(tmp / "cache"),
        extraArgs: @[],
      )

      let started = epochTime()
      generateTalkingHead(r, "musetalk", narration, outMp4, opts)
      let dt = epochTime() - started
      echo &"  live MuseTalk render took {dt:.1f}s"

      doAssert fileExists(outMp4), "no MP4 at " & outMp4
      let sz = getFileSize(outMp4)
      echo &"  output: {sz} bytes"
      # The spec requires >50 KB. A real MuseTalk render of a few
      # seconds of audio is in the low MB range.
      check sz > 50_000

      let probe = ffprobeJson(outMp4)
      var hasVideo = false
      var hasAudio = false
      for s in probe{"streams"}.items:
        let kind = s{"codec_type"}.getStr()
        if kind == "video": hasVideo = true
        elif kind == "audio": hasAudio = true
      check hasVideo
      # MuseTalk's final ffmpeg combine step muxes the input WAV into
      # the output MP4. We rely on that documented behaviour.
      check hasAudio

      let videoDur = parseFloat(probe{"format", "duration"}.getStr())
      let narProbe = ffprobeJson(narration)
      let narDur = parseFloat(narProbe{"format", "duration"}.getStr())
      echo &"  narration dur: {narDur:.3f}s; talking-head dur: {videoDur:.3f}s"
      check abs(videoDur - narDur) <= 0.5

      # Cache hit — a second call must return without spawning the
      # subprocess again (well under 5 s; a real hit is <1 s).
      let secondStart = epochTime()
      let outMp4_2 = tmp / "live2.mp4"
      generateTalkingHead(r, "musetalk", narration, outMp4_2, opts)
      let secondDt = epochTime() - secondStart
      echo &"  second call (cache hit) took {secondDt:.3f}s"
      check secondDt < 5.0
      check fileExists(outMp4_2)
      check getFileSize(outMp4_2) == getFileSize(outMp4)

#!/usr/bin/env bash
# Build EI.app (Apple silicon) with the AI engine inside, and optionally a DMG and an install.
#
#   packaging/macos/ei/package.sh             # -> dist/EI.app
#   packaging/macos/ei/package.sh --dmg       # also dist/EI-<version>-macos-arm64.dmg
#   packaging/macos/ei/package.sh --install   # also copy to /Applications/EI.app
#   packaging/macos/ei/package.sh --skip-build
#
# Needs: Rust (rustup), Xcode command line tools. The icon is drawn by make_icon.py (Pillow:
# uses ai-engine/.venv or EI's own Python environment if present). Signing is ad-hoc: the app runs
# on this Mac; other Macs would need a Developer ID signature.
#
# The AI engine's Python packages are not bundled: on first use, AI ▸ AI Engine Setup… ▸ Install
# creates ~/Library/Application Support/EI/venv. ffmpeg comes from Homebrew (brew install ffmpeg).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
HERE="$ROOT/packaging/macos/ei"
BUILD=1; DMG=0; INSTALL=0
while [ $# -gt 0 ]; do
  case "$1" in
    --skip-build) BUILD=0 ;;
    --dmg) DMG=1 ;;
    --install) INSTALL=1 ;;
    -h|--help) sed -n '2,16p' "$0"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
  shift
done
export PATH="/opt/homebrew/opt/rustup/bin:$HOME/.cargo/bin:$PATH"
export MACOSX_DEPLOYMENT_TARGET=11.0
VERSION="$(awk '/^\[/ { p = ($0 == "[workspace.package]"); next } p && $1 == "version" { gsub(/[" ]/, "", $3); print $3; exit }' "$ROOT/Cargo.toml")"
SHA="$(git -C "$ROOT" rev-parse --short HEAD 2>/dev/null || echo unknown)"
TARGET="${CARGO_TARGET_DIR:-$ROOT/target}"
DIST="$ROOT/dist"
APP="$DIST/EI.app"

if [ "$BUILD" = 1 ]; then
  echo "==> cargo build --release (EI $VERSION, $SHA)"
  (cd "$ROOT" && FILMCRAFT_BUILD_SHA="$SHA" CARGO_BUILD_JOBS="${CARGO_BUILD_JOBS:-4}" cargo build --release --locked -p filmcraft)
fi
BIN="$TARGET/release/filmcraft"
[ -x "$BIN" ] || { echo "error: $BIN not built" >&2; exit 1; }

echo "==> assembling $APP"
rm -rf "$APP"
mkdir -p "$APP/Contents/MacOS" "$APP/Contents/Resources/ai-engine"
cp "$BIN" "$APP/Contents/MacOS/EI"

# AI engine: the Python sources and requirements only (no venv, fixtures or caches).
( cd "$ROOT/ai-engine" && cp ./*.py requirements.txt README.md "$APP/Contents/Resources/ai-engine/" )

# Icon
PY=""
for c in "$HOME/Library/Application Support/EI/venv/bin/python" "$ROOT/ai-engine/.venv/bin/python" python3; do
  if "$c" -c "import PIL" >/dev/null 2>&1; then PY="$c"; break; fi
done
if [ -n "$PY" ]; then
  "$PY" "$HERE/make_icon.py" "$APP/Contents/Resources/EI.icns"
else
  echo "warning: Pillow not found; EI.app gets the generic icon" >&2
fi

cat >"$APP/Contents/Info.plist" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>CFBundleDevelopmentRegion</key><string>en</string>
  <key>CFBundleDisplayName</key><string>EI</string>
  <key>CFBundleName</key><string>EI</string>
  <key>CFBundleExecutable</key><string>EI</string>
  <key>CFBundleIdentifier</key><string>io.github.abhsonu143.ei</string>
  <key>CFBundleInfoDictionaryVersion</key><string>6.0</string>
  <key>CFBundlePackageType</key><string>APPL</string>
  <key>CFBundleSignature</key><string>????</string>
  <key>CFBundleShortVersionString</key><string>${VERSION%%-*}</string>
  <key>CFBundleVersion</key><string>${VERSION%%-*}</string>
  <key>EIBuildCommit</key><string>$SHA</string>
  <key>CFBundleIconFile</key><string>EI</string>
  <key>LSApplicationCategoryType</key><string>public.app-category.video</string>
  <key>LSMinimumSystemVersion</key><string>11.0</string>
  <key>NSHighResolutionCapable</key><true/>
  <key>NSSupportsAutomaticGraphicsSwitching</key><true/>
  <key>NSRequiresAquaSystemAppearance</key><false/>
  <key>NSMicrophoneUsageDescription</key><string>EI records voiceovers from the microphone when you ask it to.</string>
  <key>NSHumanReadableCopyright</key><string>EI, based on FilmCraft. MIT OR Apache-2.0.</string>
  <key>UTExportedTypeDeclarations</key>
  <array>
    <dict>
      <key>UTTypeIdentifier</key><string>io.github.abhsonu143.ei.project</string>
      <key>UTTypeDescription</key><string>EI Project</string>
      <key>UTTypeConformsTo</key><array><string>public.data</string><string>public.json</string></array>
      <key>UTTypeIconFile</key><string>EI</string>
      <key>UTTypeTagSpecification</key>
      <dict><key>public.filename-extension</key><array><string>fcproj</string></array></dict>
    </dict>
  </array>
</dict>
</plist>
PLIST
plutil -lint "$APP/Contents/Info.plist" >/dev/null
printf 'APPL????' >"$APP/Contents/PkgInfo"

echo "==> ad-hoc signing"
codesign --force --sign - "$APP/Contents/MacOS/EI"
codesign --force --sign - "$APP"
codesign --verify --strict --verbose=1 "$APP"

if [ "$DMG" = 1 ]; then
  OUT="$DIST/EI-$VERSION-macos-arm64.dmg"
  STAGE="$(mktemp -d)"
  ditto "$APP" "$STAGE/EI.app"
  ln -s /Applications "$STAGE/Applications"
  rm -f "$OUT" "$STAGE.raw.dmg"
  hdiutil makehybrid -hfs -hfs-volume-name "EI" -hfs-openfolder "$STAGE" -o "$STAGE.raw.dmg" "$STAGE" >/dev/null
  hdiutil convert "$STAGE.raw.dmg" -format UDZO -imagekey zlib-level=9 -o "$OUT" >/dev/null
  rm -rf "$STAGE" "$STAGE.raw.dmg"
  echo "==> $OUT"
fi

if [ "$INSTALL" = 1 ]; then
  echo "==> installing to /Applications/EI.app"
  rm -rf /Applications/EI.app
  ditto "$APP" /Applications/EI.app
fi
echo "==> done: $APP"

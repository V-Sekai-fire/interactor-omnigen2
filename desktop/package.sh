#!/bin/sh
# Build one desktop package of OmniGen2 + EditScore7B-v1.1 (program only; the weights ship separately).
#   package.sh <osx-arm64|linux-64|win-64> <out-dir> <pixi-pack binary> <pixi-unpack binary for the target>
# Result in <out-dir>: omnigen2-editscore-<os>.zip (or .tar for Linux), files at the top level.
set -eu
PLATFORM=$1; OUT=$2; PACK=$3; UNPACK=$4
HERE=$(cd "$(dirname "$0")" && pwd)
REPO=$(cd "$HERE/.." && pwd)
case $PLATFORM in
  osx-arm64) NAME=omnigen2-editscore-macos-arm64; ARC=zip ;;
  linux-64)  NAME=omnigen2-editscore-linux-x86_64; ARC=tar ;;
  win-64)    NAME=omnigen2-editscore-windows-x86_64; ARC=zip ;;
  *) echo "unknown platform $PLATFORM" >&2; exit 2 ;;
esac
STAGE="$OUT/$NAME"
rm -rf "$STAGE"; mkdir -p "$STAGE/app"
"$PACK" pack --manifest-path "$HERE/pixi.toml" --platform "$PLATFORM" --output-file "$STAGE/environment.tar"
cp "$HERE/omnigen2_edit.py" "$STAGE/app/"
# Inference code only: no training, data or optimiser modules.
(cd "$REPO" && tar cf - --exclude='__pycache__' omnigen2/__init__.py omnigen2/models omnigen2/pipelines \
   omnigen2/schedulers omnigen2/utils omnigen2/ops omnigen2/cache_functions omnigen2/taylorseer_utils) | (cd "$STAGE/app" && tar xf -)
cp "$REPO/LICENSE" "$STAGE/LICENSE"
cp "$HERE/README.txt" "$STAGE/README.txt"
if [ "$PLATFORM" = win-64 ]; then
  cp "$UNPACK" "$STAGE/pixi-unpack.exe"; cp "$HERE/omnigen2-editscore.cmd" "$STAGE/"
else
  cp "$UNPACK" "$STAGE/pixi-unpack"; chmod +x "$STAGE/pixi-unpack"
  cp "$HERE/omnigen2-editscore" "$STAGE/"; chmod +x "$STAGE/omnigen2-editscore"
fi
rm -f "$OUT/$NAME.$ARC"
if [ "$ARC" = zip ]; then
  (cd "$STAGE" && zip -q -r -0 "../$NAME.zip" .)
else
  (cd "$STAGE" && tar cf "../$NAME.tar" .)
fi
ls -l "$OUT/$NAME.$ARC"

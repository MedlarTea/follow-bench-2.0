#!/usr/bin/env bash
# Download, verify and unpack the FollowBench 2.0 release assets (custom CARLA
# build + model weights), then check the result with check_install.py.
#
# Usage:
#   bash tools/install/install_assets.sh [options] [PACKAGE ...]
#
# PACKAGE: carla | weights | oa_vat | trackvla | core (= carla weights, default) | all
#   weights also fetches YOLO11 and oa_vat also fetches YOLOe from their upstream
#   publishers (AGPL-3.0, not redistributed in the zips).
#
# Options:
#   --download         Download missing zips from the Hugging Face dataset
#                      hjyeee/Follow-Bench-2.0-packages (resumable). Without it, the
#                      zips must already be in --zip-dir.
#   --zip-dir DIR      Where the zips are / are downloaded to (default: <repo>/downloads).
#   --carla-dir DIR    Install location of the CARLA build (default: <repo>/CARLA_multi_robots,
#                      which every script uses when CARLA_ROOT / CARLA_BIN are unset).
#   --install-wheel    pip-install the CARLA client wheel shipped with the build into $PYTHON.
#   --delete-zips      Delete each zip after it has been verified and unpacked.
#   --force            Re-extract even if the package is already installed.
#   --check            Only run tools/install/check_install.py for these packages.
#   -h, --help         Show this help.
#
# Environment: HF_ENDPOINT (e.g. a mirror), FB_ASSETS_REPO, FB_ASSETS_REVISION, PYTHON.
set -euo pipefail

usage() { sed -n '2,25p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; }

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
MANIFEST="$SCRIPT_DIR/assets.tsv"
FILES="$SCRIPT_DIR/files.tsv"
PYTHON="${PYTHON:-python}"
command -v "$PYTHON" >/dev/null || PYTHON=python3
HF_ENDPOINT="${HF_ENDPOINT:-https://huggingface.co}"
HF_ENDPOINT="${HF_ENDPOINT%/}"
FB_ASSETS_REPO="${FB_ASSETS_REPO:-hjyeee/Follow-Bench-2.0-packages}"
FB_ASSETS_REVISION="${FB_ASSETS_REVISION:-main}"

ZIP_DIR="$REPO_ROOT/downloads"
CARLA_DIR="$REPO_ROOT/CARLA_multi_robots"
DOWNLOAD=0 DELETE_ZIPS=0 FORCE=0 INSTALL_WHEEL=0 CHECK_ONLY=0
REQUESTED=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --zip-dir)       ZIP_DIR="${2:?}"; shift 2 ;;
    --carla-dir)     CARLA_DIR="${2:?}"; shift 2 ;;
    --download)      DOWNLOAD=1; shift ;;
    --install-wheel) INSTALL_WHEEL=1; shift ;;
    --delete-zips)   DELETE_ZIPS=1; shift ;;
    --force)         FORCE=1; shift ;;
    --check)         CHECK_ONLY=1; shift ;;
    -h|--help)       usage; exit 0 ;;
    -*)              echo "[assets] unknown option: $1" >&2; usage >&2; exit 2 ;;
    *)               REQUESTED+=("$1"); shift ;;
  esac
done
[[ ${#REQUESTED[@]} -gt 0 ]] || REQUESTED=(core)

mkdir -p "$(dirname "$CARLA_DIR")"
CARLA_DIR="$(cd "$(dirname "$CARLA_DIR")" && pwd)/$(basename "$CARLA_DIR")"

# ── Manifest ──────────────────────────────────────────────────────────────────
declare -A F_FILE F_SHA F_ZBYTES F_UBYTES F_DEST F_MARKER
ALL_PKGS=()
while IFS=$'\t' read -r name file sha zbytes ubytes dest marker _desc; do
  [[ -z "$name" || "$name" == \#* ]] && continue
  ALL_PKGS+=("$name")
  F_FILE[$name]="$file"; F_SHA[$name]="$sha"; F_ZBYTES[$name]="$zbytes"
  F_UBYTES[$name]="$ubytes"; F_DEST[$name]="$dest"; F_MARKER[$name]="$marker"
done <"$MANIFEST"

PKGS=()
for p in "${REQUESTED[@]}"; do
  case "$p" in
    core) PKGS+=(carla weights) ;;
    all)  PKGS+=("${ALL_PKGS[@]}") ;;
    *)    [[ -n "${F_FILE[$p]:-}" ]] || { echo "[assets] unknown package: $p (choose: ${ALL_PKGS[*]} core all)" >&2; exit 2; }
          PKGS+=("$p") ;;
  esac
done
declare -A SEEN; UNIQ=()
for p in "${PKGS[@]}"; do [[ -n "${SEEN[$p]:-}" ]] || { SEEN[$p]=1; UNIQ+=("$p"); }; done
PKGS=("${UNIQ[@]}")

# check_install.py scopes that correspond to the requested packages
SCOPES=()
for p in "${PKGS[@]}"; do
  case "$p" in carla|weights) [[ " ${SCOPES[*]-} " == *" core "* ]] || SCOPES+=(core) ;; *) SCOPES+=("$p") ;; esac
done
run_check() { "$PYTHON" "$SCRIPT_DIR/check_install.py" --carla-dir "$CARLA_DIR" "$@" "${SCOPES[@]}"; }

if (( CHECK_ONLY )); then run_check; exit $?; fi

mkdir -p "$ZIP_DIR"
ZIP_DIR="$(cd "$ZIP_DIR" && pwd)"
for t in unzip sha256sum curl; do
  command -v "$t" >/dev/null || { echo "[assets] '$t' is required" >&2; exit 2; }
done

target_root() { [[ "${F_DEST[$1]}" == carla ]] && echo "$CARLA_DIR" || echo "$REPO_ROOT"; }
is_installed() { [[ -e "$(target_root "$1")/${F_MARKER[$1]}" ]]; }
file_size() { stat -c %s "$1" 2>/dev/null || echo 0; }
gb() { awk -v b="$1" 'BEGIN{printf "%.1f GB", b/1e9}'; }
zip_url() { echo "$HF_ENDPOINT/datasets/$FB_ASSETS_REPO/resolve/$FB_ASSETS_REVISION/$1"; }

# curl with resume; $1 url, $2 destination, $3 expected bytes
download() {
  local url="$1" dest="$2" want="$3" part="$2.part"
  if [[ "$(file_size "$part")" != "$want" ]]; then
    curl -fL --connect-timeout 20 --retry 5 --retry-delay 5 -C - -o "$part" "$url" || return 1
  fi
  local got; got="$(file_size "$part")"
  if [[ "$got" != "$want" ]]; then
    echo "[assets] $(basename "$dest"): got $got bytes, expected $want; re-run to resume" >&2
    return 1
  fi
  mv "$part" "$dest"
}

check_space() {  # pkg
  local p="$1" need=0 avail dir
  is_installed "$p" && (( ! FORCE )) && return 0
  [[ -f "$ZIP_DIR/${F_FILE[$p]}" ]] || need=$(( need + ${F_ZBYTES[$p]} ))
  need=$(( need + ${F_UBYTES[$p]} ))
  dir="$(target_root "$p")"; [[ -d "$dir" ]] || dir="$(dirname "$dir")"
  avail=$(( $(df -Pk "$dir" | awk 'NR==2{print $4}') * 1024 ))
  if (( avail < need )); then
    echo "[assets] $p: needs about $(gb $need) free on $(df -P "$dir" | awk 'NR==2{print $6}'), only $(gb $avail) available" >&2
    return 1
  fi
}

fetch() {  # pkg -> ensures the zip is in ZIP_DIR
  local p="$1" zip="$ZIP_DIR/${F_FILE[$1]}"
  [[ -f "$zip" ]] && return 0
  if (( ! DOWNLOAD )); then
    echo "[assets] $p: $zip not found. Re-run with --download, or download it yourself:" >&2
    echo "         $(zip_url "${F_FILE[$p]}")" >&2
    return 1
  fi
  echo "[assets] $p: downloading ${F_FILE[$p]} ($(gb "${F_ZBYTES[$p]}")) from $HF_ENDPOINT/datasets/$FB_ASSETS_REPO"
  download "$(zip_url "${F_FILE[$p]}")" "$zip" "${F_ZBYTES[$p]}"
}

verify() {  # pkg
  local p="$1" zip="$ZIP_DIR/${F_FILE[$1]}" got
  if [[ "$(file_size "$zip")" != "${F_ZBYTES[$p]}" ]]; then
    echo "[assets] $p: $zip is $(file_size "$zip") bytes, expected ${F_ZBYTES[$p]}; delete it and download again" >&2
    return 1
  fi
  echo "[assets] $p: verifying SHA256 ..."
  got="$(sha256sum "$zip" | cut -d' ' -f1)"
  if [[ "$got" != "${F_SHA[$p]}" ]]; then
    echo "[assets] $p: checksum mismatch (got $got, want ${F_SHA[$p]}); delete $zip and download again" >&2
    return 1
  fi
}

extract() {  # pkg
  local p="$1" zip="$ZIP_DIR/${F_FILE[$1]}" tmp
  echo "[assets] $p: extracting -> $(target_root "$p")"
  if [[ "${F_DEST[$p]}" == carla ]]; then
    tmp="$(mktemp -d "$(dirname "$CARLA_DIR")/.fb_carla_XXXXXX")"
    unzip -q "$zip" -d "$tmp"
    rm -rf "$CARLA_DIR"
    mv "$tmp/CARLA_multi_robots" "$CARLA_DIR"
    rmdir "$tmp"
    chmod +x "$CARLA_DIR/CarlaUE4.sh" "$CARLA_DIR/CarlaUE4/Binaries/Linux/CarlaUE4-Linux-Shipping"
  else
    mkdir -p "$REPO_ROOT/data"
    unzip -q -o "$zip" -d "$REPO_ROOT"
  fi
  is_installed "$p" || { echo "[assets] $p: extraction finished but ${F_MARKER[$p]} is missing" >&2; return 1; }
  (( DELETE_ZIPS )) && rm -f "$zip"
  return 0
}

# Upstream (non-redistributed, publicly downloadable) files of a group in files.tsv.
fetch_upstream() {  # group
  local group="$1" rc=0 _g size sha path sources src url dest ok
  while IFS=$'\t' read -r _g size sha path sources; do
    dest="$REPO_ROOT/$path"
    if [[ -f "$dest" && "$(file_size "$dest")" == "$size" ]] && [[ "$(sha256sum "$dest" | cut -d' ' -f1)" == "$sha" ]]; then
      echo "[assets] $group: $(basename "$dest") already present"; continue
    fi
    mkdir -p "$(dirname "$dest")"; rm -f "$dest"
    ok=0
    for src in $sources; do
      case "$src" in
        hf:*)  url="$HF_ENDPOINT/$(cut -d/ -f1-2 <<<"${src#hf:}")/resolve/main/$(cut -d/ -f3- <<<"${src#hf:}")" ;;
        url:*) url="${src#url:}" ;;
        *)     continue ;;
      esac
      echo "[assets] $group: downloading $(basename "$dest") from $url"
      if download "$url" "$dest" "$size" && [[ "$(sha256sum "$dest" | cut -d' ' -f1)" == "$sha" ]]; then
        ok=1; break
      fi
      echo "[assets] $group: $url failed or checksum mismatch, trying next source" >&2
      rm -f "$dest" "$dest.part"
    done
    (( ok )) || { echo "[assets] $group: could not download $path" >&2; rc=1; }
  done < <(awk -F'\t' -v g="$group" '$1==g' "$FILES")
  return $rc
}

echo "[assets] packages: ${PKGS[*]}"
echo "[assets] zip dir:  $ZIP_DIR"
FAILED=()
for p in "${PKGS[@]}"; do
  if is_installed "$p" && (( ! FORCE )); then
    echo "[assets] $p: already installed at $(target_root "$p") (use --force to reinstall)"
  elif check_space "$p" && fetch "$p" && verify "$p" && extract "$p"; then
    echo "[assets] $p: ok"
  else
    FAILED+=("$p"); continue
  fi
  case "$p" in
    weights) echo "[assets] YOLO11 is AGPL-3.0 (https://github.com/ultralytics/ultralytics/blob/main/LICENSE)"
             fetch_upstream yolo || FAILED+=(yolo) ;;
    oa_vat)  echo "[assets] YOLOe is AGPL-3.0 (https://github.com/THU-MIG/yoloe)"
             fetch_upstream yoloe || FAILED+=(yoloe) ;;
  esac
done

if [[ " ${PKGS[*]} " == *" carla "* ]] && [[ -x "$CARLA_DIR/CarlaUE4.sh" ]] && (( INSTALL_WHEEL )); then
  whl="$(ls "$CARLA_DIR"/PythonAPI/carla/dist/carla-*-cp310-*.whl 2>/dev/null | head -n1)"
  echo "[assets] installing CARLA client wheel into $("$PYTHON" -c 'import sys; print(sys.executable)'): $whl"
  "$PYTHON" -m pip install --force-reinstall --no-deps "$whl"
fi

echo
echo "[assets] license notes:"
for p in "${PKGS[@]}"; do
  case "$p" in
    weights) echo "  KPR ReID: Hippocratic License 3.0, read data/weights/reid_kpr/LICENSE" ;;
    oa_vat)  echo "  MobileCLIP-B-LT: non-commercial research only, read data/oa-vat/LICENSE_MODELS" ;;
  esac
done
for p in "${PKGS[@]}"; do
  case "$p" in
    oa_vat)   echo "[assets] OA-VAT also needs DINOv3 ViT-B/16 (gated, not redistributable): docs/01_install.md#dinov3-for-oa-vat" ;;
    trackvla) echo "[assets] TrackVLA also needs the DINOv3 ViT-S/16 snapshot (gated): docs/01_install.md#dinov3-for-trackvla" ;;
  esac
done

echo
check_args=(-q)
(( INSTALL_WHEEL )) || check_args+=(--no-python)
run_check "${check_args[@]}" || true

if [[ ${#FAILED[@]} -gt 0 ]]; then
  echo "[assets] FAILED: ${FAILED[*]}" >&2
  exit 1
fi
echo "[assets] done."
if [[ -x "$CARLA_DIR/CarlaUE4.sh" ]]; then
  [[ "$CARLA_DIR" == "$REPO_ROOT/CARLA_multi_robots" ]] || echo "  export CARLA_ROOT=$CARLA_DIR"
  (( INSTALL_WHEEL )) || echo "  next: pip install --force-reinstall --no-deps $CARLA_DIR/PythonAPI/carla/dist/carla-0.9.16-cp310-cp310-linux_x86_64.whl"
fi

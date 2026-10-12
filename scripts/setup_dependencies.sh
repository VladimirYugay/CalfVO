#!/bin/bash
# Clones croco and dust3r at the commits this work was evaluated against, and
# patches croco's absolute imports so it can be imported as a package.
#
# Usage:  bash scripts/setup_dependencies.sh [--with-curope]
#
#   --with-curope   also compile the CUDA RoPE2D kernel. Without it croco falls
#                   back to a slower RoPE that gives the same answer.
set -euo pipefail

CROCO_COMMIT=743ee71a2a9bf57cea6832a9064a70a0597fcfcb
DUST3R_COMMIT=c9e9336a6ba7c1f1873f9295852cea6dffaf770d

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WITH_CUROPE=0
[ "${1:-}" = "--with-curope" ] && WITH_CUROPE=1

clone_pinned() {
    local url=$1 commit=$2 dest=$3
    if [ -e "$dest" ]; then
        echo "==> $dest already exists, leaving it alone"
        return
    fi
    echo "==> cloning $url at $commit"
    git clone --quiet "$url" "$dest"
    git -C "$dest" checkout --quiet "$commit"
    # Drop the nested history so this repo does not end up with a repo inside it.
    rm -rf "$dest/.git"
}

clone_pinned https://github.com/naver/croco.git "$CROCO_COMMIT" "$REPO_ROOT/croco"
clone_pinned https://github.com/naver/dust3r.git "$DUST3R_COMMIT" "$REPO_ROOT/dust3r_repo"

# dust3r's importable package is the dust3r/ subdirectory of its repository.
if [ ! -e "$REPO_ROOT/dust3r" ]; then
    echo "==> installing the dust3r package at $REPO_ROOT/dust3r"
    mv "$REPO_ROOT/dust3r_repo/dust3r" "$REPO_ROOT/dust3r"
fi
rm -rf "$REPO_ROOT/dust3r_repo"

# Apply the import patch, unless it is already in.
if grep -q '^from models.blocks import' "$REPO_ROOT/croco/models/croco.py"; then
    echo "==> patching croco/models/croco.py for package-relative imports"
    (cd "$REPO_ROOT/croco" && patch -p1 --forward \
        < "$REPO_ROOT/scripts/croco-relative-imports.patch")
else
    echo "==> croco/models/croco.py already patched"
fi

if [ "$WITH_CUROPE" = 1 ]; then
    echo "==> building the CUDA RoPE2D kernel"
    (cd "$REPO_ROOT/croco/models/curope" && python setup.py build_ext --inplace)
fi

echo
echo "Done. croco and dust3r are in place; both are CC BY-NC-SA 4.0 (Naver),"
echo "non-commercial use only. See their own LICENSE files."

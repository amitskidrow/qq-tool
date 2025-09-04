#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

timestamp() {
  date +"%Y-%m-%d %H:%M:%S"
}

# Bump patch version in pyproject.toml and src/qq/__init__.py
bump_version() {
  local pyproject="$ROOT_DIR/pyproject.toml"
  local init_py="$ROOT_DIR/src/qq/__init__.py"
  if [[ ! -f "$pyproject" ]]; then
    echo "pyproject.toml not found; skipping version bump"
    return 0
  fi
  local current
  current=$(sed -nE 's/^version = "([^"]+)"/\1/p' "$pyproject" | head -n1)
  if [[ -z "$current" ]]; then
    echo "Could not determine current version from pyproject.toml; skipping version bump"
    return 0
  fi
  IFS='.' read -r MAJOR MINOR PATCH <<< "$current"
  if [[ -z "${MAJOR:-}" || -z "${MINOR:-}" || -z "${PATCH:-}" ]]; then
    echo "Unexpected version format: $current; skipping version bump"
    return 0
  fi
  local next_patch=$((PATCH + 1))
  local next_version="${MAJOR}.${MINOR}.${next_patch}"
  # Update pyproject
  sed -i -E 's/^(version = ")[0-9]+\.[0-9]+\.[0-9]+(")/\1'"${next_version}"'\2/' "$pyproject"
  # Update package __version__ for CLI --version
  if [[ -f "$init_py" ]]; then
    sed -i -E 's/^(__version__ = ")[0-9]+\.[0-9]+\.[0-9]+(")/\1'"${next_version}"'\2/' "$init_py"
  fi
  echo "Bumped version: ${current} -> ${next_version}"
}

commit_and_push() {
  local ts
  ts=$(timestamp)
  git -C "$ROOT_DIR" add -A
  if ! git -C "$ROOT_DIR" diff --staged --quiet; then
    local v
    v=$(sed -nE 's/^version = "([^"]+)"/\1/p' "$ROOT_DIR/pyproject.toml" | head -n1 || true)
    if [[ -n "$v" ]]; then
      git -C "$ROOT_DIR" commit -m "chore(release): v${v} (${ts})"
    else
      git -C "$ROOT_DIR" commit -m "Auto deploy: ${ts}"
    fi
    echo "Committed changes"
  else
    echo "No changes to commit"
  fi
  # Push current branch (works for feature branches and main)
  local branch
  branch=$(git -C "$ROOT_DIR" rev-parse --abbrev-ref HEAD 2>/dev/null || echo "main")
  git -C "$ROOT_DIR" push -u origin "$branch" || true
}

# Convert origin URL to a PEP 508 VCS spec suitable for uvx/pipx
origin_spec() {
  local origin
  origin=$(git -C "$ROOT_DIR" remote get-url origin 2>/dev/null || echo "")
  if [[ -z "$origin" ]]; then
    echo ""; return 0
  fi
  # Prefer explicit deploy branch, defaulting to api_sql_arch for this migration
  local branch
  branch=${DEPLOY_BRANCH:-api_sql_arch}
  if [[ -z "$branch" ]]; then
    branch=$(git -C "$ROOT_DIR" rev-parse --abbrev-ref HEAD 2>/dev/null || echo "main")
  fi
  local https
  if [[ "$origin" =~ ^git@github.com:(.*)\.git$ ]]; then
    https="https://github.com/${BASH_REMATCH[1]}.git"
  elif [[ "$origin" =~ ^ssh://git@github.com/(.*)\.git$ ]]; then
    https="https://github.com/${BASH_REMATCH[1]}.git"
  else
    https="$origin"
  fi
  if [[ "$https" != git+* ]]; then
    https="git+${https}"
  fi
  # default to @<branch> ref
  if [[ "$https" != *"@"* ]]; then
    https="${https}@${branch}"
  fi
  echo "$https"
}

install_remote() {
  echo "Waiting for remote to be ready..."
  sleep 15

  local spec
  spec=$(origin_spec)

  if command -v uvx >/dev/null 2>&1; then
    echo "Checking via uvx from $spec"
    uvx --from "$spec" qq --version || echo "uvx qq --version failed"
  fi

  # Prefer global install with uv tool so 'which qq' resolves
  if command -v uv >/dev/null 2>&1; then
    echo "Installing globally via: uv tool install --force --from $spec qq"
    if UV_TOOL_OUT=$(uv tool install --force --from "$spec" qq 2>&1); then
      echo "$UV_TOOL_OUT" | sed -n '1,80p'
      echo "qq --version -> $(qq --version 2>&1 || true)"
      echo "which qq -> $(command -v qq || echo not found)"
      return 0
    else
      echo -e "uv tool install failed, output follows:\n$UV_TOOL_OUT"
    fi
  fi

  # Fallback to pipx if available (support both new and old pipx)
  if command -v pipx >/dev/null 2>&1; then
    echo "Installing globally via: pipx install --force --spec $spec qq (or legacy syntax)"
    if PIPX_OUT=$(pipx install --force --spec "$spec" qq 2>&1); then
      echo "$PIPX_OUT" | sed -n '1,80p'
      echo "qq --version -> $(qq --version 2>&1 || true)"
      echo "which qq -> $(command -v qq || echo not found)"
      return 0
    else
      # Legacy pipx (no --spec); try positional VCS spec
      echo "pipx --spec not supported; trying: pipx install --force $spec"
      if PIPX_OUT2=$(pipx install --force "$spec" 2>&1); then
        echo "$PIPX_OUT2" | sed -n '1,80p'
        echo "qq --version -> $(qq --version 2>&1 || true)"
        echo "which qq -> $(command -v qq || echo not found)"
        return 0
      else
        echo -e "pipx install failed, output follows:\n$PIPX_OUT\n--- legacy attempt ---\n$PIPX_OUT2"
      fi
    fi
  fi

  # Final fallback: install from local checkout
  if command -v uv >/dev/null 2>&1; then
    echo "Falling back to local install via uv tool"
    if UV_TOOL_OUT2=$(uv tool install --force --from . qq 2>&1); then
      echo "$UV_TOOL_OUT2" | sed -n '1,80p'
      echo "qq --version -> $(qq --version 2>&1 || true)"
      echo "which qq -> $(command -v qq || echo not found)"
      return 0
    fi
  fi
  if command -v pipx >/dev/null 2>&1; then
    echo "Falling back to local install via pipx"
    if PIPX_OUT3=$(pipx install --force . 2>&1); then
      echo "$PIPX_OUT3" | sed -n '1,80p'
      echo "qq --version -> $(qq --version 2>&1 || true)"
      echo "which qq -> $(command -v qq || echo not found)"
      return 0
    fi
  fi
  echo "No usable installer or all installs failed; performed only ephemeral uvx check"
}

main() {
  # Allow skipping the bump via env: BUMP=0 ./deploy.sh
  local do_bump
  do_bump=${BUMP:-1}
  if [[ "$do_bump" == "1" || "$do_bump" == "true" ]]; then
    bump_version
  else
    echo "Skipping version bump (BUMP=$do_bump)"
  fi
  commit_and_push
  install_remote
}

main "$@"

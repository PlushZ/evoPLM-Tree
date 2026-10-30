#!/usr/bin/env bash
# Pre-submission check. Run from the repository root before sbatch:
#
#   bash cluster/preflight.sh                                  # environment only
#   bash cluster/preflight.sh cluster/slurm/controls/foo.sh    # also check that launcher
#
# Catches the failures that otherwise only surface minutes later in a job log:
# a conda environment that does not exist, the package not importable, dead
# data symlinks, and a launcher whose inputs are missing.

set -uo pipefail

fail=0
ok()   { printf '  \033[32mOK\033[0m    %s\n' "$1"; }
bad()  { printf '  \033[31mFAIL\033[0m  %s\n' "$1"; fail=1; }
warn() { printf '  \033[33mWARN\033[0m  %s\n' "$1"; }

echo "== repository =="
if [[ -f pyproject.toml && -d cluster/slurm ]]; then
  ok "running from repository root ($PWD)"
else
  bad "not the repository root — cd to it before submitting"
  exit 1
fi

echo "== conda environment =="
if [[ -f cluster/env.local.sh ]]; then
  # shellcheck disable=SC1091
  source cluster/env.local.sh
  ok "cluster/env.local.sh found (CONDA_ENV=${CONDA_ENV:-unset})"
else
  warn "no cluster/env.local.sh — launchers will use the published default 'evoplm'"
fi
CONDA_ENV="${CONDA_ENV:-evoplm}"

if source "$HOME/miniconda/etc/profile.d/conda.sh" 2>/dev/null; then
  ok "conda initialised"
  if conda env list | awk '{print $1}' | grep -qx "$CONDA_ENV"; then
    ok "environment '$CONDA_ENV' exists"
    PY="$(conda info --base 2>/dev/null)/envs/$CONDA_ENV/bin/python"
    if [[ -x "$PY" ]]; then
      if "$PY" -c "import evoplm, torch, transformers, pandas, scipy" 2>/dev/null; then
        ok "evoplm, torch, transformers, pandas, scipy all import"
      else
        bad "imports failed in '$CONDA_ENV' — try: conda run -n $CONDA_ENV pip install -e . --no-deps"
      fi
    else
      bad "no python in '$CONDA_ENV'"
    fi
  else
    bad "environment '$CONDA_ENV' not found. Available: $(conda env list | awk 'NR>2 {printf "%s ", $1}')"
  fi
else
  bad "could not source conda profile"
fi

echo "== data links =="
broken=0
while IFS= read -r link; do
  [[ -e "$link" ]] || { bad "dead link: $link"; broken=1; }
done < <(find data models -maxdepth 4 -type l 2>/dev/null)
[[ $broken -eq 0 ]] && ok "all symlinks under data/ and models/ resolve"
if [[ -z "$(find data models -maxdepth 4 -type l 2>/dev/null)" ]]; then
  warn "no symlinks found — run: bash cluster/setup_data_links.sh"
fi

if [[ $# -ge 1 ]]; then
  launcher="$1"
  echo "== launcher: $launcher =="
  if [[ ! -f "$launcher" ]]; then
    bad "no such file"
  else
    bash -n "$launcher" && ok "bash syntax" || bad "bash syntax"

    # Inputs only. results/ is where jobs write, so absence there is expected.
    missing=0
    while IFS= read -r path; do
      [[ -e "$path" ]] || { bad "missing input: $path"; missing=1; }
    done < <(grep -oE '(data|models|configs)/[A-Za-z0-9_./-]+' "$launcher" \
             | grep -vE '\$|\*' | sort -u)
    [[ $missing -eq 0 ]] && ok "referenced data/model/config paths exist"

    # Compute nodes can ship an older /lib64/libstdc++ than the conda env's
    # extensions were built against, so a launcher that activates an env must
    # also put that env's lib directory on the loader path. The login node
    # often has a new enough system library, so this cannot be caught by
    # importing here — it has to be a static check on the script.
    if grep -q 'conda activate' "$launcher"; then
      if grep -q 'LD_LIBRARY_PATH.*CONDA_PREFIX' "$launcher"; then
        ok "exports LD_LIBRARY_PATH from CONDA_PREFIX"
      else
        bad "no LD_LIBRARY_PATH export — pandas/scipy will fail on nodes with an older libstdc++"
      fi
    fi

    if grep -q 'BASH_SOURCE' "$launcher" && ! grep -q 'SLURM_SUBMIT_DIR' "$launcher"; then
      bad "resolves its path from BASH_SOURCE without SLURM_SUBMIT_DIR — breaks under sbatch"
    fi

    if grep -q 'gres=gpu' "$launcher"; then
      echo "  note: GPU job — queue time depends on availability"
    fi
  fi
fi

echo
if [[ $fail -eq 0 ]]; then
  echo "preflight passed — safe to sbatch"
else
  echo "preflight FAILED — fix the above before submitting"
fi
exit $fail

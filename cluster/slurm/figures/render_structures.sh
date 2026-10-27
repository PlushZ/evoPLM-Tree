#!/usr/bin/env bash
# Render all PyMOL .pml files in a directory to PNGs.
# Run:
#   sbatch cluster/slurm/figures/render_structures.sh

#SBATCH -J pymol_render
#SBATCH -p cpu-single
#SBATCH -N 1 -c 2 --mem=16G
#SBATCH -t 01:00:00
#SBATCH --export=NONE
#SBATCH -o logs/render_all_pml_remapped.out
#SBATCH -e logs/render_all_pml_remapped.err

set -euo pipefail
export PATH="$HOME/bin:/usr/bin:/bin"
source "$HOME/miniconda/etc/profile.d/conda.sh"
[[ -f cluster/env.local.sh ]] && source cluster/env.local.sh
conda activate "${CONDA_ENV:-evoplm}"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
mkdir -p logs

# ------------
# User config 
# ------------
PML_GLOB="results/generation/split1/hotspots_to_structure/*.pml"
RAY_W=2400
RAY_H=1800
DPI=300
BG_COLOR="white"
TRANSPARENT_BG=0
# If 1, enforce shared min/max color scale per pred/true pair.
USE_PAIR_SHARED_SCALE=1
# If 1, apply percentile clipping for non-count (frequency) files.
# This preserves shared pred/true scaling, but boosts contrast for small differences.
USE_PERCENTILE_CLIP_FOR_FREQ=1
# Percentile bounds used when clipping frequency-mode scales.
SCALE_LOW_PCT=0
SCALE_HIGH_PCT=98
# Color scheme used in PyMOL spectrum command.
COLOR_SCHEME="blue_white_red"
# If 1, save a separate legend image per rendered structure PNG.
SAVE_COLORBAR_PNG=1
COLORBAR_WIDTH_IN=1.8
COLORBAR_HEIGHT_IN=4.2
# If 1, enforce strict frequency-like colors for non-count files.
# Useful when B-factors were written as frequency * FREQ_BFACTOR_SCALE.
STRICT_FREQ_COLORS=1
FREQ_BFACTOR_SCALE=100
STRICT_FREQ_MIN=0
# Use 0..0.2 as the full blue->white->red gradient; values >0.2 saturate at red.
STRICT_FREQ_MAX=0.01
# Rotation (degrees) applied after orient/zoom and before ray.
ROTATE_X_DEG=270 #90
ROTATE_Y_DEG=0
ROTATE_Z_DEG=270
# Camera shift after rotation (Angstrom along view axis; negative moves object closer).
MOVE_Z_ANG=0
# For RBD files only (filename starts with "rbd_"): color only RBD, keep the rest gray.
RBD_ONLY_HIGHLIGHT=1
RBD_GREY_COLOR="gray60"
# Selection used for 6VSB RBD highlighting.
RBD_SELECTION_6VSB="chain A and resi 319-541"
# Selection used for 6M0J RBD highlighting.
RBD_SELECTION_6M0J="chain E"

# Optional: set to a specific PyMOL executable path/command.
# Leave empty for auto-detection.
PYMOL_BIN=""

find_pymol_bin() {
  if [[ -n "$PYMOL_BIN" ]]; then
    if command -v "$PYMOL_BIN" >/dev/null 2>&1; then
      echo "$PYMOL_BIN"
      return 0
    fi
    echo "[error] Configured PYMOL_BIN is not executable in PATH: $PYMOL_BIN" >&2
    return 1
  fi

  if command -v pymol >/dev/null 2>&1; then
    echo "pymol"
    return 0
  fi
  if command -v pymol-open-source >/dev/null 2>&1; then
    echo "pymol-open-source"
    return 0
  fi
  if command -v mamba >/dev/null 2>&1; then
    if mamba run -n pymol pymol -cq -d "quit" >/dev/null 2>&1; then
      echo "mamba run -n pymol pymol"
      return 0
    fi
  fi
  if command -v conda >/dev/null 2>&1; then
    if conda run -n pymol pymol -cq -d "quit" >/dev/null 2>&1; then
      echo "conda run -n pymol pymol"
      return 0
    fi
  fi
  return 1
}

if ! PYMOL_CMD="$(find_pymol_bin)"; then
  echo "[error] Could not find a PyMOL executable." >&2
  echo "[hint] Options to fix:" >&2
  echo "  1) Install into current env: conda activate evoplm && conda install -c conda-forge pymol-open-source" >&2
  echo "  2) Use dedicated env and set PYMOL_BIN, e.g. PYMOL_BIN='conda run -n pymol pymol'" >&2
  echo "  3) If your HPC uses modules, load one before submit (e.g., module load pymol)." >&2
  exit 1
fi

shopt -s nullglob
files=( $PML_GLOB )
shopt -u nullglob

if [[ ${#files[@]} -eq 0 ]]; then
  echo "[error] No .pml files matched: $PML_GLOB" >&2
  exit 1
fi

echo "[info] Found ${#files[@]} PML files"
echo "[info] Render settings: ${RAY_W}x${RAY_H}, dpi=${DPI}, bg=${BG_COLOR}, transparent=${TRANSPARENT_BG}"
echo "[info] Using PyMOL command: ${PYMOL_CMD}"
echo "[info] Rotation: x=${ROTATE_X_DEG}, y=${ROTATE_Y_DEG}, z=${ROTATE_Z_DEG}"
echo "[info] Camera move: z=${MOVE_Z_ANG} A"
echo "[info] Shared pred/true scale: ${USE_PAIR_SHARED_SCALE}"
echo "[info] Percentile clip for frequency mode: ${USE_PERCENTILE_CLIP_FOR_FREQ} (${SCALE_LOW_PCT}-${SCALE_HIGH_PCT} pct)"
echo "[info] Color scheme: ${COLOR_SCHEME}"
echo "[info] Save colorbar PNG: ${SAVE_COLORBAR_PNG}"

declare -A PAIR_MIN
declare -A PAIR_MAX
declare -A PAIR_PMIN
declare -A PAIR_PMAX

map_pymol_to_mpl_cmap() {
  local scheme="$1"
  case "$scheme" in
    blue_white_red) echo "bwr" ;;
    red_white_blue) echo "bwr_r" ;;
    blue_red) echo "coolwarm" ;;
    rainbow) echo "rainbow" ;;
    rainbow2) echo "gist_rainbow" ;;
    green_white_magenta) echo "PiYG_r" ;;
    yellow_orange_red) echo "YlOrRd" ;;
    cyan_white_magenta) echo "cool" ;;
    *) echo "bwr" ;;
  esac
}

save_colorbar_png() {
  local out_png="$1"
  local minv="$2"
  local maxv="$3"
  local cmap="$4"
  local cbar_png="${out_png%.png}_colorscale.png"

  python - "$cbar_png" "$minv" "$maxv" "$cmap" "$COLORBAR_WIDTH_IN" "$COLORBAR_HEIGHT_IN" <<'PY'
import sys
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.colorbar import ColorbarBase

out_png, minv, maxv, cmap, w, h = sys.argv[1:7]
minv = float(minv)
maxv = float(maxv)
w = float(w)
h = float(h)

fig, ax = plt.subplots(figsize=(w, h))
norm = Normalize(vmin=minv, vmax=maxv)
cb = ColorbarBase(ax, cmap=plt.get_cmap(cmap), norm=norm, orientation="vertical")
cb.set_label("Mutation frequency")
fig.tight_layout()
fig.savefig(out_png, dpi=300)
plt.close(fig)
PY
  echo "[info] Saved colorbar: ${cbar_png}"
}

MPL_CMAP="$(map_pymol_to_mpl_cmap "$COLOR_SCHEME")"

pair_key_from_pml() {
  local p="$1"
  local bn
  bn="$(basename "$p")"
  if [[ "$bn" == *_pred.pml ]]; then
    echo "${p%_pred.pml}"
  elif [[ "$bn" == *_true.pml ]]; then
    echo "${p%_true.pml}"
  else
    echo "${p%.pml}"
  fi
}

extract_chain_from_name() {
  local bn="$1"
  if [[ "$bn" =~ _chain([A-Za-z0-9])_ ]]; then
    echo "${BASH_REMATCH[1]}"
  else
    echo ""
  fi
}

resolve_bfactor_for_pair() {
  local pair_base="$1"
  local kind="$2" # pred|true
  local candidate="${pair_base}_${kind}_bfactor.pdb"
  if [[ -f "$candidate" ]]; then
    echo "$candidate"
    return 0
  fi
  local base_dir base_name found
  base_dir="$(dirname "$pair_base")"
  base_name="$(basename "$pair_base")"
  found="$(find "$base_dir" -maxdepth 4 -type f -name "${base_name}_${kind}_bfactor.pdb" -print -quit 2>/dev/null || true)"
  if [[ -n "$found" && -f "$found" ]]; then
    echo "$found"
    return 0
  fi
  return 1
}

compute_pair_minmax() {
  local pair_base="$1"
  local pred_pdb true_pdb
  pred_pdb="$(resolve_bfactor_for_pair "$pair_base" "pred" || true)"
  true_pdb="$(resolve_bfactor_for_pair "$pair_base" "true" || true)"
  if [[ -z "$pred_pdb" || -z "$true_pdb" || ! -f "$pred_pdb" || ! -f "$true_pdb" ]]; then
    return 1
  fi

  awk '
    ($1=="ATOM" || $1=="HETATM") {
      v = substr($0, 61, 6) + 0.0
      if (!init || v < minv) minv = v
      if (!init || v > maxv) maxv = v
      init = 1
    }
    END {
      if (!init) exit 1
      printf("%.3f %.3f\n", minv, maxv)
    }
  ' "$pred_pdb" "$true_pdb"
}

compute_pair_percentile_range() {
  local pair_base="$1"
  local low_pct="$2"
  local high_pct="$3"
  local pred_pdb true_pdb
  pred_pdb="$(resolve_bfactor_for_pair "$pair_base" "pred" || true)"
  true_pdb="$(resolve_bfactor_for_pair "$pair_base" "true" || true)"
  if [[ -z "$pred_pdb" || -z "$true_pdb" || ! -f "$pred_pdb" || ! -f "$true_pdb" ]]; then
    return 1
  fi

  awk '
    ($1=="ATOM" || $1=="HETATM") { print substr($0, 61, 6) + 0.0 }
  ' "$pred_pdb" "$true_pdb" \
    | sort -n \
    | awk -v lo="$low_pct" -v hi="$high_pct" '
      { a[NR] = $1 }
      END {
        n = NR
        if (n < 1) exit 1
        if (lo < 0) lo = 0
        if (hi > 100) hi = 100
        if (hi < lo) { t = lo; lo = hi; hi = t }
        li = int((lo / 100.0) * (n - 1)) + 1
        hii = int((hi / 100.0) * (n - 1)) + 1
        if (li < 1) li = 1
        if (hii < 1) hii = 1
        if (li > n) li = n
        if (hii > n) hii = n
        printf("%.3f %.3f\n", a[li], a[hii])
      }
    '
}

for pml in "${files[@]}"; do
  RUN_PML="$pml"
  TMP_PML=""
  out_png="${pml%.pml}.png"
  bn="$(basename "$pml")"
  EXTRA_CMDS=""
  SPECTRUM_SELECTION="all"
  FILE_CHAIN="$(extract_chain_from_name "$bn")"
  if [[ -n "$FILE_CHAIN" ]]; then
    SPECTRUM_SELECTION="chain ${FILE_CHAIN}"
  fi
  PAIR_KEY="$(pair_key_from_pml "$pml")"
  SCALE_MIN=""
  SCALE_MAX=""
  COLORBAR_MIN=""
  COLORBAR_MAX=""

  if [[ "${RBD_ONLY_HIGHLIGHT}" == "1" && "$bn" == rbd_* ]]; then
    # Make whole protein gray, then recolor only the desired RBD region by B-factor.
    EXTRA_CMDS+="color ${RBD_GREY_COLOR}, all; "
    if [[ "$bn" == *6VSB* ]]; then
      SPECTRUM_SELECTION="${RBD_SELECTION_6VSB}"
    elif [[ "$bn" == *6M0J* ]]; then
      SPECTRUM_SELECTION="${RBD_SELECTION_6M0J}"
    fi
    echo "[info] RBD-only highlight applied for: $bn"
  fi

  if [[ "${USE_PAIR_SHARED_SCALE}" == "1" ]]; then
    if [[ -z "${PAIR_MIN[$PAIR_KEY]+x}" ]]; then
      if mm="$(compute_pair_minmax "$PAIR_KEY")"; then
        PAIR_MIN[$PAIR_KEY]="${mm%% *}"
        PAIR_MAX[$PAIR_KEY]="${mm##* }"
      else
        PAIR_MIN[$PAIR_KEY]=""
        PAIR_MAX[$PAIR_KEY]=""
      fi
    fi
    SCALE_MIN="${PAIR_MIN[$PAIR_KEY]}"
    SCALE_MAX="${PAIR_MAX[$PAIR_KEY]}"

    # For frequency-mode files (not *_count_*), optionally clip by percentile.
    if [[ "${USE_PERCENTILE_CLIP_FOR_FREQ}" == "1" && "$bn" != *_count_* ]]; then
      if [[ -z "${PAIR_PMIN[$PAIR_KEY]+x}" ]]; then
        if pmm="$(compute_pair_percentile_range "$PAIR_KEY" "$SCALE_LOW_PCT" "$SCALE_HIGH_PCT")"; then
          PAIR_PMIN[$PAIR_KEY]="${pmm%% *}"
          PAIR_PMAX[$PAIR_KEY]="${pmm##* }"
        else
          PAIR_PMIN[$PAIR_KEY]=""
          PAIR_PMAX[$PAIR_KEY]=""
        fi
      fi
      if [[ -n "${PAIR_PMIN[$PAIR_KEY]}" && -n "${PAIR_PMAX[$PAIR_KEY]}" ]]; then
        SCALE_MIN="${PAIR_PMIN[$PAIR_KEY]}"
        SCALE_MAX="${PAIR_PMAX[$PAIR_KEY]}"
      fi
    fi

    if [[ -n "$SCALE_MIN" && -n "$SCALE_MAX" ]]; then
      EXTRA_CMDS+="spectrum b, ${COLOR_SCHEME}, (${SPECTRUM_SELECTION}), minimum=${SCALE_MIN}, maximum=${SCALE_MAX}; "
      COLORBAR_MIN="${SCALE_MIN}"
      COLORBAR_MAX="${SCALE_MAX}"
      if [[ "${USE_PERCENTILE_CLIP_FOR_FREQ}" == "1" && "$bn" != *_count_* && -n "${PAIR_PMIN[$PAIR_KEY]}" ]]; then
        echo "[info] Shared clipped scale for pair $(basename "$PAIR_KEY"): min=${SCALE_MIN}, max=${SCALE_MAX} (pct ${SCALE_LOW_PCT}-${SCALE_HIGH_PCT})"
      else
        echo "[info] Shared scale for pair $(basename "$PAIR_KEY"): min=${SCALE_MIN}, max=${SCALE_MAX}"
      fi
    else
      echo "[warn] Could not compute shared scale for pair $(basename "$PAIR_KEY"), using PML default."
    fi
  fi

  # Strict frequency color mode (non-count files):
  # Avoid PyMOL alter-in-one-line parsing issues: keep B-values unchanged and map
  # the strict frequency interval onto [0, FREQ_BFACTOR_SCALE] render range.
  if [[ "${STRICT_FREQ_COLORS}" == "1" && "$bn" != *_count_* ]]; then
    SCALE_MIN="$(awk -v x="${STRICT_FREQ_MIN}" -v s="${FREQ_BFACTOR_SCALE}" 'BEGIN{printf "%.6f", x*s}')"
    SCALE_MAX="$(awk -v x="${STRICT_FREQ_MAX}" -v s="${FREQ_BFACTOR_SCALE}" 'BEGIN{printf "%.6f", x*s}')"
    COLORBAR_MIN="${STRICT_FREQ_MIN}"
    COLORBAR_MAX="${STRICT_FREQ_MAX}"
    EXTRA_CMDS+="spectrum b, ${COLOR_SCHEME}, (${SPECTRUM_SELECTION}), minimum=${SCALE_MIN}, maximum=${SCALE_MAX}; "
    echo "[info] Strict frequency colors for $bn: render_scale=${SCALE_MIN}..${SCALE_MAX}, colorbar_scale=${COLORBAR_MIN}..${COLORBAR_MAX}"
  fi

  EXTRA_CMDS+="turn x, ${ROTATE_X_DEG}; turn y, ${ROTATE_Y_DEG}; turn z, ${ROTATE_Z_DEG}; move z, ${MOVE_Z_ANG}; "

  # If a PML contains a stale absolute load path (e.g., directory renamed),
  # rewrite the load target to the current PML directory for this render.
  LOAD_PATH="$(awk -F',' '/^load[[:space:]]+/ {sub(/^load[[:space:]]+/, "", $1); print $1; exit}' "$pml" || true)"
  if [[ -n "$LOAD_PATH" && ! -f "$LOAD_PATH" ]]; then
    FALLBACK_PATH="$(dirname "$pml")/$(basename "$LOAD_PATH")"
    if [[ ! -f "$FALLBACK_PATH" ]]; then
      FOUND_FALLBACK="$(find "$(dirname "$pml")" -maxdepth 4 -type f -name "$(basename "$LOAD_PATH")" -print -quit 2>/dev/null || true)"
      if [[ -n "$FOUND_FALLBACK" && -f "$FOUND_FALLBACK" ]]; then
        FALLBACK_PATH="$FOUND_FALLBACK"
      fi
    fi
    if [[ -f "$FALLBACK_PATH" ]]; then
      TMP_PML="$(mktemp --suffix=.pml)"
      sed -E "s|^load[[:space:]]+[^,]+,|load ${FALLBACK_PATH},|" "$pml" > "$TMP_PML"
      RUN_PML="$TMP_PML"
      echo "[warn] Rewriting stale load path for $(basename "$pml")"
      echo "       old: $LOAD_PATH"
      echo "       new: $FALLBACK_PATH"
    else
      echo "[warn] PML load target not found and no fallback file exists:"
      echo "       missing: $LOAD_PATH"
      echo "       tried:   $FALLBACK_PATH"
    fi
  fi

  echo "[info] Rendering: $RUN_PML -> $out_png"
  ${PYMOL_CMD} -cq "$RUN_PML" -d "bg_color ${BG_COLOR}; set ray_opaque_background, ${TRANSPARENT_BG}; orient; zoom; ${EXTRA_CMDS}ray ${RAY_W},${RAY_H}; png ${out_png}, dpi=${DPI}; quit"
  if [[ "${SAVE_COLORBAR_PNG}" == "1" && -n "${COLORBAR_MIN}" && -n "${COLORBAR_MAX}" ]]; then
    save_colorbar_png "$out_png" "$COLORBAR_MIN" "$COLORBAR_MAX" "$MPL_CMAP"
  fi
  [[ -n "$TMP_PML" ]] && rm -f "$TMP_PML"
done

echo "[ok] Completed rendering all PML files."

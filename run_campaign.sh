#!/usr/bin/env bash
# Solve, validate and certify the published newblk matrix end to end.
#
#   bash run_campaign.sh                 # full campaign, published EU blocks
#   bash run_campaign.sh --blocks rule   # alternative EU boundary assignment
#   bash run_campaign.sh --from 2        # resume at step 2 (workbooks/results kept)
#   bash run_campaign.sh --dry-run       # print the plan and check preconditions
#
# Any failing step aborts the campaign. results/CAMPAIGN_DONE is written only
# after every solve, welfare file, validation and build has succeeded,
# and records what produced it.
#
# NOTE: bash reads a script file incrementally. Never edit this file while a
# campaign is running -- doing so desynchronises the interpreter and produces a
# spurious syntax error mid-run. Copy it to a new name and edit the copy.

set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

BLOCKS=csv
FROM=0
DRY=0
while [ $# -gt 0 ]; do
  case "$1" in
    --blocks) BLOCKS="${2:-}"; shift 2;;
    --from)   FROM="${2:-0}";  shift 2;;
    --dry-run) DRY=1; shift;;
    -h|--help) sed -n '2,15p' "$0"; exit 0;;
    *) echo "unknown argument: $1" >&2; exit 2;;
  esac
done

# Workbook directory: override with CAMPAIGN_WORKBOOKS. The default is the
# build directory of the SAM reconstruction; it must exist and hold the
# variants this campaign deploys.
WORKBOOKS="${CAMPAIGN_WORKBOOKS:-replication/workbooks}"
PY=".venv/Scripts/python.exe"
SYS_PY="python"
SENTINEL="results/CAMPAIGN_DONE"
LOCK="results/.campaign.lock"
export PYTHONIOENCODING=utf-8 PYTHONUTF8=1

log() { echo "=== [$(date '+%m-%d %H:%M:%S')] $* ==="; }
die() { echo "FATAL: $*" >&2; exit 1; }

case "$BLOCKS" in
  csv)  CENTRAL_SRC=newblk_eul;      WB_MAIN="";                      WB_CHN35="$WORKBOOKS/csv35.xlsx";;
  rule) CENTRAL_SRC=newblk_rule_eul; WB_MAIN="$WORKBOOKS/rule50.xlsx"; WB_CHN35="$WORKBOOKS/rule35.xlsx";;
  *) die "--blocks must be csv or rule (got '$BLOCKS')";;
esac

# The 16 families the manuscript reports. The final certification requires all
# of them explicitly: --available would pass while silently omitting absentees.
PUBLISHED_FAMILIES=(
  newblk newblk_k25 newblk_k50 newblk_k75
  sens_newblk_sigma_en_low sens_newblk_sigma_en_high
  sens_newblk_armington_low sens_newblk_armington_high
  sens_newblk_carbon_75 sens_newblk_carbon_125
  sens_newblk_cbam_direct_only
  sens_newblk_tstar_2028 sens_newblk_tstar_2030
  sens_newblk_adj_cost_low sens_newblk_adj_cost_high sens_newblk_adj_cost_higher
)

# ---- preconditions --------------------------------------------------------
log "preconditions (blocks=$BLOCKS, from step $FROM)"
[ -x "$PY" ] || die "model interpreter not found: $PY"
[ -f sinretencion.py ] || die "sinretencion.py not found"
[ -d "$WORKBOOKS" ] || die "workbook directory not found: $WORKBOOKS (set CAMPAIGN_WORKBOOKS)"
[ -f "$WB_CHN35" ] || die "missing workbook: $WB_CHN35"
[ -z "$WB_MAIN" ] || [ -f "$WB_MAIN" ] || die "missing workbook: $WB_MAIN"
for f in results_3d results_2d results_4d parameters sam_data; do
  [ -s "results/${f}_${CENTRAL_SRC}.parquet" ] || die "central source missing: results/${f}_${CENTRAL_SRC}.parquet"
done
git diff --quiet -- exiobase3_2022_10x9.xlsx || die "the committed workbook has uncommitted edits; restore it first"

if [ "$DRY" = 1 ]; then
  log "dry run: preconditions passed"
  echo "  central source : $CENTRAL_SRC"
  echo "  workbooks      : $WORKBOOKS"
  echo "  families       : ${#PUBLISHED_FAMILIES[@]}"
  exit 0
fi

# Single-instance lock: two concurrent campaigns race on the same suffixes and
# on the shared workbook file (this bit once).
if ! ( set -o noclobber; echo "$$ $(date)" > "$LOCK" ) 2>/dev/null; then
  die "another campaign is running (lock $LOCK holds: $(cat "$LOCK" 2>/dev/null))"
fi
cleanup() {
  rm -f "$LOCK"
  # Never leave a non-canonical workbook deployed.
  git checkout -- exiobase3_2022_10x9.xlsx 2>/dev/null || true
}
trap cleanup EXIT

rm -f "$SENTINEL"

# ---- 0. promote the central family ---------------------------------------
if [ "$FROM" -le 0 ]; then
  log "0. promote $CENTRAL_SRC -> newblk"
  for f in results_3d results_2d results_4d parameters sam_data; do
    cp "results/${f}_${CENTRAL_SRC}.parquet" "results/${f}_newblk.parquet"
  done
  if [ -n "$WB_MAIN" ]; then
    cp "$WB_MAIN" exiobase3_2022_10x9.xlsx
    log "deployed $WB_MAIN as the canonical workbook"
  fi
fi

# ---- 1. welfare, kappa sweep, sensitivity grid -----------------------------
if [ "$FROM" -le 1 ]; then
  log "1. welfare + kappa sweep + sensitivity paper plan"
  "$PY" recenter_compute.py --skip-central --force
fi

# ---- 2. China-35% robustness ----------------------------------------------
if [ "$FROM" -le 2 ]; then
  log "2. China-35% robustness"
  cp "$WB_CHN35" exiobase3_2022_10x9.xlsx
  "$PY" run_central_eul.py newblk_chn35
  "$PY" compute_welfare.py --suffix newblk_chn35 > results/logs/welfare_newblk_chn35.log 2>&1
  if [ -n "$WB_MAIN" ]; then cp "$WB_MAIN" exiobase3_2022_10x9.xlsx; else git checkout -- exiobase3_2022_10x9.xlsx; fi
  log "canonical workbook restored"
fi

# ---- 3. certify every published family explicitly --------------------------
if [ "$FROM" -le 3 ]; then
  log "3. validate the ${#PUBLISHED_FAMILIES[@]} published families"
  args=()
  for fam in "${PUBLISHED_FAMILIES[@]}"; do
    [ -s "results/parameters_${fam}.parquet" ] || die "family missing from the matrix: $fam"
    args+=(--suffix "$fam")
  done
  "$PY" validate_batch_results.py "${args[@]}" > results/logs/validate_campaign.log 2>&1 \
    || { tail -20 results/logs/validate_campaign.log >&2; die "validation failed"; }
  log "validation passed (see results/logs/validate_campaign.log)"
fi

# ---- 4. paper assets ------------------------------------------------------
if [ "$FROM" -le 4 ]; then
  log "4. build paper assets"
  (
    cd new_version_paper/cbam_paper_updated/code
    PAPER_RUN_SUFFIX=newblk $SYS_PY build.py > build_campaign.log 2>&1 \
      || { tail -20 build_campaign.log >&2; exit 1; }
  ) || die "build failed"
  log "build passed"
fi

# ---- sentinel with provenance ---------------------------------------------
{
  echo "campaign: blocks=$BLOCKS central_source=$CENTRAL_SRC"
  echo "finished: $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
  echo "commit:   $(git rev-parse HEAD)"
  echo "dirty:    $(git status --porcelain | wc -l) tracked/untracked entries"
  echo "families: ${PUBLISHED_FAMILIES[*]}"
} > "$SENTINEL"

log "CAMPAIGN DONE -> $SENTINEL"

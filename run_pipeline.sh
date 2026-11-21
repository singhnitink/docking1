#!/usr/bin/env bash
# In-container driver. Two ways to use it:
#
#   MODE 1 (fold + score)  protein sequence + ligand SMILES
#                          -> OpenFold3 co-folds ONE structure
#                          -> AutoDock Vina scores the predicted pose
#
#   MODE 2 (score only)    an existing protein-ligand CIF
#                          -> AutoDock Vina scores it
#                          Works with AlphaFold3, OpenFold3 and Boltz output.
#
# Run via:  apptainer run --nv docking_of3.sif [options]
set -euo pipefail

# Debugging aid: RUN_PIPELINE_TRACE=/some/file traces every command the driver
# runs, with timestamps, to that file (kept apart from the tools' own output).
if [[ -n "${RUN_PIPELINE_TRACE:-}" ]]; then
    exec 7>>"$RUN_PIPELINE_TRACE"
    BASH_XTRACEFD=7
    PS4='+ $(date +%s) ${BASH_LINENO[0]}: '
    set -x
fi

DOCKING_HOME="${DOCKING_HOME:-/opt/docking}"
OF3_PY="${OF3_PY:-/opt/conda/envs/of3/bin}"
DOCK_PY="${DOCK_PY:-/opt/conda/envs/docking/bin/python}"
OF3_CKPT="${OF3_CKPT:-/opt/openfold3/of3-p2-155k.pt}"   # baked into the image, if any

NAME=""
PROTEIN=""        # either the sequence itself or a path to a FASTA file
QUERY_JSON=""
CIF_INPUT=""
LIGAND_CHAIN=""
CHECKPOINT=""
OUTDIR="docking_output"
MSA_SERVER=true
USE_TEMPLATES=""
FOLD_ONLY=false
SETUP_WEIGHTS=false
SMILES=()
DOCK_ARGS=()
MODE1_FLAGS=()    # Mode-1-only flags seen, to reject them in Mode 2

# Every checkpoint in openfold3's registry that works with this version, best first.
OF3_CKPT_NAMES=(of3-p2-155k.pt of3-p2-145k.pt)
OF3_CKPT_DEFAULT_NAME="openfold3-p2-155k"
# A real checkpoint is ~2.3 GB; anything much smaller is a truncated download.
OF3_CKPT_MIN_BYTES=1000000000

# Parameters are looked for here, in order: --checkpoint, baked into the image,
# then the user cache. OPENFOLD_CACHE must be writable - openfold3 writes a
# ckpt_root marker into it - so it defaults to $HOME, which Apptainer mounts
# from the host.
: "${OPENFOLD_CACHE:=${HOME:-/tmp}/.openfold3}"
case "$OPENFOLD_CACHE" in
    /*) ;;
    *)  OPENFOLD_CACHE="$PWD/$OPENFOLD_CACHE" ;;   # never a relative path: it would
esac                                               # land wherever the caller stands
export OPENFOLD_CACHE

usage() {
    cat <<'EOF'
OpenFold3 co-folding + AutoDock Vina binding scores.

MODE 1 - fold a complex, then score it            (needs --nv on the apptainer command)
  --fasta SEQ|FILE                the protein: the sequence itself, or a path
                                  to a FASTA file (--sequence is an alias)
  --smiles SMILES                 the ligand; repeat for several ligands, and
                                  each one is then scored against the complex
  --name NAME                     job name (default: the FASTA filename, else "job")
  --query-json FILE               use a ready-made OpenFold3 query JSON instead
  --fold-only                     stop after folding, do not score
  --no-msa-server                 single-sequence mode (no internet, less accurate)
  --use-templates                 let OpenFold3 use structural templates
  --checkpoint FILE               OpenFold3 parameters to use (default: found
                                  automatically, see SETUP)

  Exactly ONE structure is produced (1 model seed, 1 diffusion sample).

MODE 2 - score a structure you already have        (no GPU, no network needed)
  --cif FILE|DIR                  protein-ligand CIF, or a directory to search
  --smiles SMILES                 ligand SMILES (only if it cannot be auto-detected)
  --ligand-chain ID               ligand chain (only if it cannot be auto-detected;
                                  default: every ligand chain is scored)

  The ligand SMILES is normally found automatically: AlphaFold3 stores it in the
  CIF, OpenFold3 in its query JSON, Boltz in its input YAML.

SETUP
  --setup-weights                 download the OpenFold3 parameters once
                                  (~2.3 GB) into $OPENFOLD_CACHE, then exit.
                                  Not needed if they are baked into the image.

COMMON
  --output-dir DIR                where results go (default: docking_output)
  -- ...                          everything after a bare -- goes to docking.py
                                  (e.g. -- --dock --exhaustiveness 32)

Inside the container only $HOME, the working directory and /tmp are visible.
Inputs or outputs anywhere else need an explicit  --bind /that/path  on the
apptainer command.

EXAMPLES
  # fold GSH onto a protein, then score it - sequence given directly
  apptainer run --nv docking_of3.sif --name gsh \
      --fasta MNPISRSFSHRIIGSRSLDSVLFSPREARSTFPDLENFRIAKLHGFRRVFAHSAPIFF \
      --smiles 'SC[C@@H](C(=O)NCC(=O)O)NC(=O)CC[C@@H](C(=O)O)N'

  # the same thing from a FASTA file
  apptainer run --nv docking_of3.sif --fasta protein.fa --smiles 'CCO'

  # score an existing complex (no GPU, no network needed)
  apptainer run docking_of3.sif --cif my_model.cif

  # score a whole output tree, and redock as well
  apptainer run docking_of3.sif --cif af3_output/ -- --dock
EOF
}

die() { echo "$*" >&2; exit "${EXIT_CODE:-2}"; }

# A flag that takes a value must be followed by one - not by the next flag, and
# not by nothing. Catching that here keeps a typo from silently shifting the
# whole command line by one.
val() {
    if [[ -z "${2-}" || "${2-}" == -* ]]; then
        die "$1 needs a value"
    fi
}

# Print the checkpoint to use, or nothing if none is available yet.
# Honours ckpt_root exactly as openfold3 does, so both agree on the file.
resolve_ckpt() {
    if [[ -n "$CHECKPOINT" ]]; then printf '%s' "$CHECKPOINT"; return; fi
    if [[ -s "$OF3_CKPT" ]]; then printf '%s' "$OF3_CKPT"; return; fi
    local root="$OPENFOLD_CACHE" r n
    if [[ -s "$OPENFOLD_CACHE/ckpt_root" ]]; then
        r="$(tr -d '[:space:]' < "$OPENFOLD_CACHE/ckpt_root")"
        [[ -d "$r" ]] && root="$r"
    fi
    for n in "${OF3_CKPT_NAMES[@]}"; do
        if [[ -s "$root/$n" ]]; then printf '%s' "$root/$n"; return; fi
    done
}

ckpt_sane() {   # a checkpoint file that is far too small is a truncated download
    local f="$1" sz
    sz=$(stat -L -c %s "$f" 2>/dev/null || echo 0)   # -L: a symlinked checkpoint is fine
    [[ "$sz" -ge "$OF3_CKPT_MIN_BYTES" ]]
}
ckpt_size() { du -hL "$1" 2>/dev/null | cut -f1; }

in_container() { [[ -n "${APPTAINER_CONTAINER:-}${SINGULARITY_CONTAINER:-}" ]]; }

# Is this path somewhere that vanishes when the container exits? Anything whose
# mount point is the container root is the image or its writable overlay (the
# fstype varies: overlay, squashfs, fuse.fuse-overlayfs, ...), and a tmpfs is
# what Apptainer puts at $HOME under --containall. A bound host directory always
# has a mount point of its own, so it never matches.
EPHEMERAL_DESC=""
ephemeral_fs() {
    in_container || return 1
    local tgt fs size
    read -r tgt fs size < <(findmnt -n -b -o TARGET,FSTYPE,SIZE -T "$1" 2>/dev/null || true)
    [[ -z "$fs" ]] && return 1
    EPHEMERAL_DESC="$fs mounted at $tgt"
    [[ "$tgt" == "/" ]] && return 0
    # Apptainer's own scratch tmpfs (what --containall puts at $HOME and /tmp) is
    # tiny: its "sessiondir max size", 64 MB by default. A host tmpfs bound in
    # (some machines keep /tmp in RAM) is many gigabytes, and is a real place.
    if [[ "$fs" == "tmpfs" && -n "${size:-}" && "$size" -le $((1024 * 1024 * 1024)) ]]; then
        EPHEMERAL_DESC="$fs mounted at $tgt, only $((size / 1024 / 1024)) MB - the container's own scratch, not a host directory"
        return 0
    fi
    return 1
}

# Refuse to write results anywhere that vanishes when the container exits.
check_outdir_persists() {
    ephemeral_fs "$1" || return 0
    cat >&2 <<EOF
--output-dir $1
is inside the container's own filesystem ($EPHEMERAL_DESC), not on the host.
Anything written there is thrown away when the container exits, so this run
would report success and leave nothing behind.

Point --output-dir at a host directory that is visible inside the container: your
home, the directory you ran apptainer from, /tmp, or anything you add with
    apptainer run --bind /some/host/dir ...
EOF
    EXIT_CODE=1 die "refusing to write into an ephemeral location"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --name)           val --name "${2-}";        NAME="$2"; shift 2 ;;
        --fasta|--sequence)
                          val "$1" "${2-}";          PROTEIN="$2"; MODE1_FLAGS+=("$1"); shift 2 ;;
        --smiles)         val --smiles "${2-}";      SMILES+=("$2"); shift 2 ;;
        --query-json)     val --query-json "${2-}";  QUERY_JSON="$2"; MODE1_FLAGS+=("$1"); shift 2 ;;
        --cif)            val --cif "${2-}";         CIF_INPUT="$2"; shift 2 ;;
        --ligand-chain|--ligand_chain)
                          val --ligand-chain "${2-}"; LIGAND_CHAIN="$2"; shift 2 ;;
        --checkpoint)     val --checkpoint "${2-}";  CHECKPOINT="$2"; MODE1_FLAGS+=("$1"); shift 2 ;;
        --output-dir)     val --output-dir "${2-}";  OUTDIR="$2"; shift 2 ;;
        --fold-only)      FOLD_ONLY=true; MODE1_FLAGS+=("$1"); shift ;;
        --setup-weights)  SETUP_WEIGHTS=true; shift ;;
        --no-msa-server)  MSA_SERVER=false; MODE1_FLAGS+=("$1"); shift ;;
        --use-templates)  USE_TEMPLATES=true; MODE1_FLAGS+=("$1"); shift ;;
        -h|--help)        usage; exit 0 ;;
        --)               shift; DOCK_ARGS=("$@"); break ;;
        *) echo "unknown option: $1" >&2; echo >&2; usage >&2; exit 2 ;;
    esac
done

# ---------------------------------------------------------------------------
# one-time parameter download
# ---------------------------------------------------------------------------
if [[ "$SETUP_WEIGHTS" == "true" ]]; then
    if [[ -s "$OF3_CKPT" ]]; then
        echo "parameters are already baked into this image: $OF3_CKPT"
        echo "nothing to do."
        exit 0
    fi
    if ! mkdir -p "$OPENFOLD_CACHE" 2>/dev/null || [[ ! -w "$OPENFOLD_CACHE" ]]; then
        cat >&2 <<EOF
cannot write to \$OPENFOLD_CACHE ($OPENFOLD_CACHE).
Bind a writable host directory and point the cache at it, e.g.
    apptainer run --bind /scratch/of3:/scratch/of3 \\
        --env OPENFOLD_CACHE=/scratch/of3 docking_of3.sif --setup-weights
EOF
        exit 1
    fi
    if ephemeral_fs "$OPENFOLD_CACHE"; then
        EXIT_CODE=1 die "\$OPENFOLD_CACHE ($OPENFOLD_CACHE) is inside the container ($EPHEMERAL_DESC); it would be lost on exit. Bind a host directory and set OPENFOLD_CACHE to it."
    fi
    echo "downloading OpenFold3 parameters into $OPENFOLD_CACHE (~2.3 GB, public, no credentials)..."
    # openfold3's own `setup_openfold` ignores \$OPENFOLD_CACHE (it writes to
    # ~/.openfold3 unconditionally) and also tries to refresh a data file inside
    # the read-only image, so call the download routine it uses directly.
    if ! "$OF3_PY/python" - "$OPENFOLD_CACHE" "$OF3_CKPT_DEFAULT_NAME" <<'PY'
import sys
from pathlib import Path
from openfold3.entry_points.parameters import download_model_parameters
d = Path(sys.argv[1])
download_model_parameters(d, sys.argv[2], skip_confirmation=True)
(d / "ckpt_root").write_text(str(d) + "\n")
PY
    then
        EXIT_CODE=1 die "download failed (see the error above). Is outbound internet available?"
    fi
    CK="$(resolve_ckpt)"
    if [[ -n "$CK" ]] && ckpt_sane "$CK"; then
        echo "done: $CK ($(ckpt_size "$CK"))"
        exit 0
    fi
    for n in "${OF3_CKPT_NAMES[@]}"; do
        if [[ -e "$OPENFOLD_CACHE/$n" ]]; then
            EXIT_CODE=1 die "$OPENFOLD_CACHE/$n exists but is only $(ckpt_size "$OPENFOLD_CACHE/$n"); a real checkpoint is ~2.3 GB. It is truncated - delete it and run --setup-weights again."
        fi
    done
    EXIT_CODE=1 die "download did not produce a checkpoint in $OPENFOLD_CACHE"
fi

# ---------------------------------------------------------------------------
# common checks
# ---------------------------------------------------------------------------
if [[ -z "$CIF_INPUT" && -z "$PROTEIN" && -z "$QUERY_JSON" ]]; then
    echo "nothing to do: give --cif to score a structure, or --fasta plus --smiles" >&2
    echo "to fold one first.  --help for details." >&2
    exit 2
fi
mkdir -p "$OUTDIR" || EXIT_CODE=1 die "cannot create --output-dir $OUTDIR"
OUTDIR="$(cd "$OUTDIR" && pwd)"       # absolute, so every printed path is real
check_outdir_persists "$OUTDIR"
SCORES="$OUTDIR/vina_scores.csv"

# ---------------------------------------------------------------------------
# MODE 2: score an existing structure
# ---------------------------------------------------------------------------
if [[ -n "$CIF_INPUT" ]]; then
    if [[ ${#MODE1_FLAGS[@]} -gt 0 ]]; then
        die "${MODE1_FLAGS[*]}: only for folding (--fasta/--smiles), cannot be combined with --cif"
    fi
    if [[ ! -e "$CIF_INPUT" ]]; then
        cat >&2 <<EOF
no such file or directory: $CIF_INPUT
(inside a container only \$HOME, the working directory and /tmp are visible;
 anything else needs  --bind /that/path  on the apptainer command)
EOF
        exit 2
    fi
    if [[ ${#SMILES[@]} -gt 1 ]]; then
        die "--cif scores one ligand definition; pass --smiles at most once"
    fi
    ARGS=("$CIF_INPUT" -w "$OUTDIR/vina_work" -o "$SCORES")
    [[ ${#SMILES[@]} -eq 1 ]] && ARGS+=(--smiles "${SMILES[0]}")
    [[ -n "$LIGAND_CHAIN" ]] && ARGS+=(--ligand_chain "$LIGAND_CHAIN")
    [[ ${#DOCK_ARGS[@]} -gt 0 ]] && ARGS+=("${DOCK_ARGS[@]}")

    echo "=== AutoDock Vina: scoring $CIF_INPUT ==="
    rc=0
    "$DOCK_PY" "$DOCKING_HOME/docking.py" "${ARGS[@]}" || rc=$?
    [[ -f "$SCORES" ]] && echo && echo "scores: $SCORES"
    exit "$rc"
fi

# ---------------------------------------------------------------------------
# MODE 1: fold, then score
# ---------------------------------------------------------------------------
if [[ -z "$QUERY_JSON" ]]; then
    if [[ ${#SMILES[@]} -eq 0 ]]; then
        die "no ligand given: pass --smiles (repeatable), or --query-json"
    fi
    # --fasta takes either the sequence itself or a path to a FASTA file. A value
    # that looks like a path but is not a file must not be folded as a "sequence".
    if [[ -f "$PROTEIN" ]]; then
        [[ -n "$NAME" ]] || NAME="$(basename "${PROTEIN%.*}")"
        PROTEIN_ARG=(--fasta "$PROTEIN")
    else
        case "$PROTEIN" in
            */*|*.fa|*.fasta|*.faa|*.fas|*.seq|*.txt|*.FA|*.FASTA)
                cat >&2 <<EOF
--fasta $PROTEIN: no such file.
(inside a container only \$HOME, the working directory and /tmp are visible;
 anything else needs  --bind /that/path  on the apptainer command)
EOF
                exit 2 ;;
        esac
        [[ -n "$NAME" ]] || NAME="job"
        PROTEIN_ARG=(--sequence "$PROTEIN")
    fi
    QUERY_JSON="$OUTDIR/${NAME}_query.json"
    MKARGS=(--name "$NAME" -o "$QUERY_JSON" "${PROTEIN_ARG[@]}")
    for s in "${SMILES[@]}"; do MKARGS+=(--smiles "$s"); done
    "$DOCK_PY" "$DOCKING_HOME/make_query.py" "${MKARGS[@]}"
else
    [[ -f "$QUERY_JSON" ]] || die "--query-json $QUERY_JSON is not a file"
    [[ -n "$NAME" ]] || NAME="$(basename "${QUERY_JSON%.*}")"
    # The scorer recovers the ligand SMILES by walking up from each CIF, so the
    # query has to live inside the output tree. A user-supplied one usually does
    # not, so keep a copy where it will be found.
    if [[ "$(cd "$(dirname "$QUERY_JSON")" && pwd)" != "$OUTDIR" ]]; then
        cp -f "$QUERY_JSON" "$OUTDIR/$(basename "$QUERY_JSON")"
    fi
fi

OF3_OUT="$OUTDIR/openfold_results"
JOB_OUT="$OF3_OUT/$NAME"
if [[ -d "$JOB_OUT" ]] && compgen -G "$JOB_OUT/*/*_model.cif" > /dev/null; then
    cat >&2 <<EOF
$JOB_OUT already holds a folded structure for a job called "$NAME".
Re-folding into it would leave two structures and make the result ambiguous.
Use a different --name or --output-dir, or remove that directory first.
EOF
    exit 1
fi
mkdir -p "$OF3_OUT"

# ---- parameters -----------------------------------------------------------
# Always hand OpenFold3 an explicit checkpoint path. Left to itself it prompts
# "Download ...? (yes/no)" on stdin and, with no terminal (a batch job, a CI
# runner), blocks for 120 s and then dies with a TimeoutError.
CKPT="$(resolve_ckpt)"
if [[ -z "$CKPT" ]]; then
    cat >&2 <<EOF

OpenFold3 parameters not found.
  looked for : $(IFS=,; echo "${OF3_CKPT_NAMES[*]}")
  in         : $OF3_CKPT  (baked into the image - absent)
               $OPENFOLD_CACHE  (\$OPENFOLD_CACHE$( [[ -s "$OPENFOLD_CACHE/ckpt_root" ]] && echo ", via ckpt_root -> $(tr -d '[:space:]' < "$OPENFOLD_CACHE/ckpt_root")"))

Fetch them once (~2.3 GB, public download, no credentials needed):

    apptainer run docking_of3.sif --setup-weights

They are stored in \$OPENFOLD_CACHE and reused by every later run. To use a
checkpoint you already have somewhere else:

    apptainer run --nv --bind /path/to/weights docking_of3.sif \\
        --checkpoint /path/to/weights/of3-p2-155k.pt ...
EOF
    exit 1
fi
[[ -f "$CKPT" ]] || EXIT_CODE=1 die "checkpoint not found: $CKPT"
if ! ckpt_sane "$CKPT"; then
    EXIT_CODE=1 die "checkpoint $CKPT is only $(ckpt_size "$CKPT"); a real one is ~2.3 GB. It is truncated or corrupt - delete it and run --setup-weights again."
fi
echo "parameters: $CKPT"

# ---- GPU preflight --------------------------------------------------------
# Check before loading 2.3 GB of weights; the usual cause is a missing --nv.
if ! "$OF3_PY/python" - <<'PY' 2>/dev/null
import torch, sys
sys.exit(0 if torch.cuda.is_available() else 1)
PY
then
    cat >&2 <<EOF
no GPU is visible inside the container, and OpenFold3 needs one.
Add  --nv  to the apptainer command:
    apptainer run --nv docking_of3.sif ...
(and check that the host's NVIDIA driver works: nvidia-smi)
EOF
    exit 1
fi

# ---- MSA server preflight -------------------------------------------------
# OpenFold3's ColabFold client never gives up: on a host with no outbound
# internet it sits in a retry loop (5 s sleeps) for as long as the job lasts.
# Check reachability once, in seconds, before anything is spent.
if [[ "$MSA_SERVER" == "true" ]]; then
    MSA_URL="${COLABFOLD_URL:-https://api.colabfold.com}"
    if ! "$OF3_PY/python" - "$MSA_URL" <<'PY' 2>/dev/null
import sys, urllib.request
try:
    urllib.request.urlopen(sys.argv[1], timeout=10)
except Exception as e:
    # an HTTP error status still proves the server answered
    sys.exit(0 if getattr(e, "code", None) else 1)
PY
    then
        cat >&2 <<EOF
cannot reach the ColabFold MSA server ($MSA_URL) from inside the container:
this host appears to have no outbound internet access. Either run on a host
that does, or use single-sequence mode (faster, less accurate, no network):

    ... --no-msa-server
EOF
        exit 1
    fi
fi

# ---- per-run scratch ------------------------------------------------------
# OpenFold3 keeps MSA scratch under \$TMPDIR/of3-of-\$USER/, one fixed path for
# the whole machine: two concurrent runs collide there, and a killed run leaves
# a directory that makes every later run refuse to start. Give each run its own
# short tmpdir (short, because sockets created under it have a ~100-char limit)
# and remove it on exit. Triton's JIT cache likewise defaults to \$HOME, which
# is not always writable in a container.
# Scratch left by a run that was killed is removed HERE, at the start of the
# next run, not by the dying run itself: when the container receives SIGTERM,
# Apptainer tears the image mount down within about a second, after which no
# external command (rm included) can execute, and a SIGKILL from a scheduler
# gives no chance at all. The PID is in the directory name so a stale one can
# be told from a live one.
for d in /tmp/of3.[0-9]*.*; do
    [[ -d "$d" ]] || continue
    pid="${d#/tmp/of3.}"; pid="${pid%%.*}"
    [[ "$pid" =~ ^[0-9]+$ ]] || continue
    kill -0 "$pid" 2>/dev/null && continue                       # still running
    if [[ "${OF3_SCRATCH_MAX_AGE_MIN:-0}" -gt 0 ]]; then
        [[ -n "$(find "$d" -maxdepth 0 -mmin +"$OF3_SCRATCH_MAX_AGE_MIN" 2>/dev/null)" ]] || continue
    fi
    rm -rf "$d" 2>/dev/null && echo "removed stale scratch $d (its run is gone)" >&2
done

RUN_TMP="$(mktemp -d "/tmp/of3.$$.XXXXXX")"
export TMPDIR="$RUN_TMP"
export TRITON_CACHE_DIR="$RUN_TMP/triton"
mkdir -p "$TRITON_CACHE_DIR"

# On a normal exit, remove the scratch. On a signal, do the one thing that
# matters and can still be done with shell builtins alone - stop OpenFold3 so
# it does not keep the GPU - and leave the directory to the sweep above.
# (The signals are converted into exits so the EXIT trap runs at all, and the
# trap disarms them on entry so a second signal cannot abort it.)
OF3_PID=""
cleanup() {
    trap '' TERM INT HUP
    set +e
    if [[ -n "$OF3_PID" ]] && kill -0 "$OF3_PID" 2>/dev/null; then
        kill -KILL -- "-$OF3_PID" 2>/dev/null || kill -KILL "$OF3_PID" 2>/dev/null
        wait "$OF3_PID" 2>/dev/null
    fi
    rm -rf "$RUN_TMP" 2>/dev/null
}
trap cleanup EXIT
# Disarm in the signal trap itself, not in cleanup: Apptainer delivers SIGTERM
# twice in quick succession, and the second one arrived before the EXIT trap
# had even started, re-entering this trap and killing the shell outright.
trap 'trap "" TERM INT HUP; exit 143' TERM
trap 'trap "" TERM INT HUP; exit 130' INT
trap 'trap "" TERM INT HUP; exit 129' HUP

# Vanilla PyTorch attention: DeepSpeed's evoformer kernel needs JIT C++/CUDA
# compilation, which does not work inside a read-only container at run time.
RUNNER_YML="$OUTDIR/runner_config.yml"
cat > "$RUNNER_YML" <<'EOF'
model_update:
  presets:
    - predict
    - low_mem
  custom:
    settings:
      memory:
        eval:
          use_deepspeed_evo_attention: false
          tune_chunk_size: false
          chunk_size: 4
EOF

# Exactly one structure: one model seed, one diffusion sample.
OF3_ARGS=(
    predict
    --query-json="$QUERY_JSON"
    --runner-yaml="$RUNNER_YML"
    --num-model-seeds=1
    --num-diffusion-samples=1
    --use-msa-server="$MSA_SERVER"
    --inference-ckpt-path="$CKPT"
    --output-dir="$OF3_OUT"
)
[[ -n "$USE_TEMPLATES" ]] && OF3_ARGS+=(--use-templates=true)

echo "=== OpenFold3: co-folding $NAME (1 seed, 1 sample) ==="
if [[ "$MSA_SERVER" == "true" ]]; then
    echo "    using the ColabFold MSA server - this needs outbound internet access"
fi
# setsid: its own process group, so cleanup can take out any workers with it
setsid "$OF3_PY/run_openfold" "${OF3_ARGS[@]}" &
OF3_PID=$!
rc=0
wait "$OF3_PID" || rc=$?
OF3_PID=""
if [[ $rc -ne 0 ]]; then
    echo "OpenFold3 exited with status $rc for $NAME; its log is above." >&2
    exit "$rc"
fi

# Confirm we got exactly the one structure we asked for - from THIS job only.
CIFS=()
while IFS= read -r line; do CIFS+=("$line"); done \
    < <(find "$JOB_OUT" -name '*_model.cif' 2>/dev/null | sort)
NCIF=${#CIFS[@]}

if [[ $NCIF -eq 0 ]]; then
    echo "OpenFold3 produced no structure for $NAME." >&2
    echo "Its own log is above; the output directory is $OF3_OUT" >&2
    exit 1
fi
if [[ $NCIF -ne 1 ]]; then
    echo "warning: expected 1 structure, found $NCIF:" >&2
    printf '  %s\n' "${CIFS[@]}" >&2
    echo "  all of them will be scored" >&2
else
    echo "structure: ${CIFS[0]}"
fi

if [[ "$FOLD_ONLY" == "true" ]]; then
    exit 0
fi

# ---------------------------------------------------------------------------
# score the predicted pose
# ---------------------------------------------------------------------------
echo
echo "=== AutoDock Vina: scoring the predicted pose ==="
ARGS=("${CIFS[@]}" -w "$OUTDIR/vina_work" -o "$SCORES")
[[ -n "$LIGAND_CHAIN" ]] && ARGS+=(--ligand_chain "$LIGAND_CHAIN")
[[ ${#DOCK_ARGS[@]} -gt 0 ]] && ARGS+=("${DOCK_ARGS[@]}")
rc=0
"$DOCK_PY" "$DOCKING_HOME/docking.py" "${ARGS[@]}" || rc=$?

echo
echo "structure : ${CIFS[0]}"
echo "scores    : $SCORES"
exit "$rc"

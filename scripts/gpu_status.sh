#!/bin/bash
# GPUクラスタの空き状況を一覧表示
# Usage:
#   bash scripts/gpu_status.sh            # 全ノード表示
#   bash scripts/gpu_status.sh --free     # 空きGPUがあるノードのみ
#   bash scripts/gpu_status.sh --me       # 自分のjobも併記

set -eu

MODE="all"
for arg in "$@"; do
    case "$arg" in
        --free) MODE="free" ;;
        --me)   MODE="me" ;;
        -h|--help)
            cat <<EOF
Usage: $0 [--free|--me]
  --free : 空きGPUがあるノードのみ表示
  --me   : 自分のjobを先頭に表示
EOF
            exit 0
            ;;
    esac
done

if [[ "$MODE" == "me" ]]; then
    echo "=== Your Jobs ==="
    squeue -u "$USER" -o "%.10i %.9P %.30j %.2t %.10M %.10L %R" || true
    echo
fi

echo "=== Node GPU Availability ==="
printf "%-12s %-8s %-8s %-8s %-8s %-10s\n" "NODE" "STATE" "TOTAL" "USED" "FREE" "FREE_MEM"
printf "%-12s %-8s %-8s %-8s %-8s %-10s\n" "----" "-----" "-----" "----" "----" "--------"

sinfo -N -h -O "NodeList:14,StateCompact:8,Gres:12,GresUsed:14,FreeMem:10" \
  | awk -v mode="$MODE" '
{
    node=$1; state=$2; gres=$3; used=$4; mem=$5
    # gres:  "gpu:8"      -> total=8
    # used:  "gpu:5(..)"  -> used=5   (括弧以降は IDX 情報)
    gsub(/gpu:/, "", gres);  total = gres + 0
    sub(/\(.*/, "", used);   gsub(/gpu:/, "", used); u = used + 0
    free = total - u
    if (state ~ /down|drain|fail/) { free = 0; state_disp=state }
    else { state_disp=state }
    if (mode == "free" && (free <= 0 || state ~ /down|drain|fail/)) next
    printf "%-12s %-8s %-8d %-8d %-8d %-10s\n", node, state_disp, total, u, free, mem
}'

echo
echo "=== Summary ==="
sinfo -N -h -O "StateCompact:8,Gres:12,GresUsed:14" \
  | awk '
{
    state=$1; gres=$2; used=$3
    gsub(/gpu:/, "", gres); total = gres + 0
    sub(/\(.*/, "", used); gsub(/gpu:/, "", used); u = used + 0
    free = total - u
    if (state ~ /down|drain|fail/) { down += total; next }
    total_all += total; used_all += u; free_all += free
    if (free >= 8) nodes_ge8++
    if (free >= 4) nodes_ge4++
    if (free >= 1) nodes_ge1++
}
END {
    printf "Total  : %d GPUs (up)\n", total_all
    printf "Used   : %d GPUs\n", used_all
    printf "Free   : %d GPUs\n", free_all
    printf "Down   : %d GPUs\n", down
    printf "Nodes with >= 8 free: %d\n", nodes_ge8+0
    printf "Nodes with >= 4 free: %d\n", nodes_ge4+0
    printf "Nodes with >= 1 free: %d\n", nodes_ge1+0
}'

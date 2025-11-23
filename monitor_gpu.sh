#!/bin/bash

# Continuous GPU monitoring script
# Monitors GPU utilization every second to catch spikes during training updates

command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Resolve NVIDIA SMI command (supports Git Bash / WSL on Windows)
NVS_CMD=""
if command_exists nvidia-smi; then
    NVS_CMD="nvidia-smi"
elif command_exists nvidia-smi.exe; then
    NVS_CMD="nvidia-smi.exe"
elif [ -x "/usr/lib/wsl/lib/nvidia-smi" ]; then
    NVS_CMD="/usr/lib/wsl/lib/nvidia-smi"
fi

if [ -z "$NVS_CMD" ]; then
    echo "❌ Unable to locate nvidia-smi. Ensure NVIDIA drivers are installed and nvidia-smi is on PATH."
    exit 1
fi

echo "🔍 Continuous GPU Monitoring (Ctrl+C to stop)"
echo "   Watch for spikes when 'TURBO MODE ACTIVE' appears in backend console"
echo "   Expected: 50-90% during updates, 5-15% during environment steps"
echo ""

while true; do
    timestamp=$(date '+%H:%M:%S')

    gpu_stats=$("$NVS_CMD" --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits 2>/dev/null)

    if [ $? -eq 0 ] && [ -n "$gpu_stats" ]; then
        while IFS=',' read -r index name utilization mem_used mem_total; do
            index=$(echo "$index" | xargs)
            name=$(echo "$name" | xargs)
            utilization=$(echo "$utilization" | xargs)
            mem_used=$(echo "$mem_used" | xargs)
            mem_total=$(echo "$mem_total" | xargs)

            if [ "$mem_total" -gt 0 ]; then
                mem_percent=$((mem_used * 100 / mem_total))
            else
                mem_percent=0
            fi

            prefix="[$timestamp] GPU${index} (${name})"
            usage_msg="Util: ${utilization}% | VRAM: ${mem_used}MB/${mem_total}MB (${mem_percent}%)"

            if [ "$utilization" -ge 80 ]; then
                echo -e "$prefix 🔥 $usage_msg ⚡ UPDATE IN PROGRESS"
            elif [ "$utilization" -ge 40 ]; then
                echo -e "$prefix 📊 $usage_msg"
            else
                echo -e "$prefix 💤 $usage_msg"
            fi
        done <<<"$gpu_stats"
    else
        echo "[$timestamp] ❌ Failed to query GPU via $NVS_CMD"
    fi

    sleep 1
done


#!/bin/bash

# Continuous GPU monitoring script
# Monitors GPU utilization every second to catch spikes during training updates

echo "🔍 Continuous GPU Monitoring (Ctrl+C to stop)"
echo "   Watch for spikes when 'TURBO MODE ACTIVE' appears in backend console"
echo "   Expected: 50-90% during updates, 5-15% during environment steps"
echo ""

while true; do
    # Get current timestamp
    timestamp=$(date '+%H:%M:%S')
    
    # Get GPU stats
    gpu_stats=$(nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits 2>/dev/null)
    
    if [ $? -eq 0 ]; then
        # Parse GPU utilization and memory
        utilization=$(echo "$gpu_stats" | cut -d',' -f1 | tr -d ' ')
        mem_used=$(echo "$gpu_stats" | cut -d',' -f2 | tr -d ' ')
        mem_total=$(echo "$gpu_stats" | cut -d',' -f3 | tr -d ' ')
        mem_percent=$((mem_used * 100 / mem_total))
        
        # Color code based on utilization
        if [ "$utilization" -ge 50 ]; then
            # High utilization - likely during update
            echo -e "[$timestamp] 🔥 GPU: ${utilization}% | Memory: ${mem_used}MB/${mem_total}MB (${mem_percent}%) ⚡ UPDATE IN PROGRESS"
        elif [ "$utilization" -ge 20 ]; then
            # Medium utilization
            echo -e "[$timestamp] 📊 GPU: ${utilization}% | Memory: ${mem_used}MB/${mem_total}MB (${mem_percent}%)"
        else
            # Low utilization - environment steps
            echo -e "[$timestamp] 💤 GPU: ${utilization}% | Memory: ${mem_used}MB/${mem_total}MB (${mem_percent}%)"
        fi
    else
        echo "[$timestamp] ❌ Failed to query GPU"
    fi
    
    sleep 1
done


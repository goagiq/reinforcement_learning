#!/bin/bash

# Manual trigger script for auto-retrain
# This bypasses file detection and triggers retraining with all files in the configured directory

echo "🔄 Triggering manual retraining..."
echo ""

response=$(curl -s -X POST http://localhost:8200/api/settings/auto-retrain/trigger-manual)

# Check if curl was successful
if [ $? -eq 0 ]; then
    echo "$response" | python -m json.tool 2>/dev/null || echo "$response"
    echo ""
    
    # Extract status from response
    status=$(echo "$response" | python -c "import sys, json; data=json.load(sys.stdin); print(data.get('status', 'unknown'))" 2>/dev/null)
    
    if [ "$status" = "success" ]; then
        echo "✅ Manual retraining triggered successfully!"
        echo "   Check the Training tab for progress"
    else
        echo "⚠️  Manual retraining may have failed. Check the response above."
    fi
else
    echo "❌ Failed to connect to backend. Is the server running on port 8200?"
    exit 1
fi


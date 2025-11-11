#!/bin/bash
# One-command runner: Downloads and runs the comparison script
# Usage: curl -s https://raw.githubusercontent.com/Arnie016/Law_Qwen/main/scripts/evaluation/run_comparison.sh | bash
# OR: Run directly: python3 <(curl -s https://raw.githubusercontent.com/Arnie016/Law_Qwen/main/scripts/evaluation/compare_base_vs_grpo.py)

SCRIPT_URL="https://raw.githubusercontent.com/Arnie016/Law_Qwen/main/scripts/evaluation/compare_base_vs_grpo.py"
SCRIPT_PATH="/tmp/compare_base_vs_grpo.py"

echo "📥 Downloading comparison script..."
curl -s "$SCRIPT_URL" -o "$SCRIPT_PATH" || {
    echo "❌ Failed to download script. Using direct Python execution..."
    python3 <(curl -s "$SCRIPT_URL")
    exit $?
}

chmod +x "$SCRIPT_PATH"
echo "✅ Script downloaded. Running comparison..."
echo ""
python3 "$SCRIPT_PATH"

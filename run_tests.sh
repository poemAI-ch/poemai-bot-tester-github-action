#!/bin/bash
# Bot testing script for GitHub Actions
# This script runs the bot tests and parses results for GitHub Actions output

set -euo pipefail

# Function to log with timestamp
log() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') $*"
}

# Function to parse test results from JSON
parse_test_results() {
    local results_file="$1"
    
    # Call the separate Python script
    python3 "$SCRIPT_DIR/parse_results.py" "$results_file"
}

# Main execution
main() {
    log "🤖 Starting automated bot testing..."
    log "  Configuration: $CONFIG_FILE"
    
    if [ -n "${PUBLISH_S3_URL:-}" ]; then
        log "  Report will be published to: $PUBLISH_S3_URL"
    fi
    echo ""
    
    # Prepare command arguments
    local args=(
        --config "$CONFIG_FILE"
    )
    
    if [ -n "${PUBLISH_S3_URL:-}" ]; then
        args+=(--publish-s3-url "$PUBLISH_S3_URL")
    fi
    
    if [ -n "${RESULTS_DIR:-}" ]; then
        args+=(--results-dir "$RESULTS_DIR")
    fi
    
    # Run the tests
    log "🚀 Executing bot tests..."
    if ! python3 "$SCRIPT_DIR/bot_tester.py" "${args[@]}"; then
        log "❌ Bot tests failed during execution"
        exit 1
    fi
    
    # Parse and output results
    parse_test_results "test_results.json"
}

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Required environment variables
: "${CONFIG_FILE:?CONFIG_FILE environment variable is required}"
: "${OPENAI_API_KEY:?OPENAI_API_KEY environment variable is required}"

# Optional environment variables with defaults
: "${RESULTS_DIR:=.}"

# Run main function
main "$@"

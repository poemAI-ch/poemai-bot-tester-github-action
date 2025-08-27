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
    
    if [ ! -f "$results_file" ]; then
        log "⚠️  Test results file not found: $results_file"
        return 1
    fi
    
    # Parse results using Python
    python3 << EOF "$results_file"
import json
import sys
import os

try:
    with open(sys.argv[1]) as f:
        data = json.load(f)
    
    # Count test results
    test_results = data.get('test_results', [])
    passed = sum(1 for r in test_results if r['test_result']['test_passed'] == 'OK')
    failed = sum(1 for r in test_results if r['test_result']['test_passed'] == 'NOK')
    skipped = sum(1 for r in test_results if r['test_result']['test_passed'] == 'SKIPPED')
    
    # Get report URL if available
    report_url = data.get('report_url', '')
    
    # Set GitHub Actions outputs
    github_output = os.environ.get('GITHUB_OUTPUT', '')
    if github_output:
        with open(github_output, 'a') as f:
            f.write(f"tests-passed={passed}\n")
            f.write(f"tests-failed={failed}\n")
            f.write(f"tests-skipped={skipped}\n")
            f.write(f"test-results-file={sys.argv[1]}\n")
            if report_url:
                f.write(f"report-url={report_url}\n")
    
    # Print summary
    print(f"📊 Test Summary:")
    print(f"  ✅ Passed: {passed}")
    print(f"  ❌ Failed: {failed}")
    print(f"  ⏭️  Skipped: {skipped}")
    
    if report_url:
        print(f"🔗 Test report published: {report_url}")
    
    # Exit with error code if there are failed tests
    if failed > 0:
        print("")
        print("❌ Some tests failed. Check the detailed results for more information.")
        sys.exit(1)
    
    print("✅ All tests completed successfully!")

except Exception as e:
    print(f"❌ Error parsing test results: {e}")
    sys.exit(1)
EOF
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

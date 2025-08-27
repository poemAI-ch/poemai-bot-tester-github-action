#!/usr/bin/env python3
"""
Parse test results and set GitHub Actions outputs
"""
import json
import os
import sys
from pathlib import Path


def parse_test_results(results_file):
    """Parse test results from JSON file and set GitHub Actions outputs"""

    if not Path(results_file).exists():
        print(f"⚠️  Test results file not found: {results_file}")
        return 1

    try:
        with open(results_file) as f:
            data = json.load(f)

        # Count test results
        test_results = data.get("test_results", [])
        passed = sum(1 for r in test_results if r["test_result"]["test_passed"] == "OK")
        failed = sum(
            1 for r in test_results if r["test_result"]["test_passed"] == "NOK"
        )
        skipped = sum(
            1 for r in test_results if r["test_result"]["test_passed"] == "SKIPPED"
        )

        # Get report URL if available
        report_url = data.get("report_url", "")

        # Set GitHub Actions outputs
        github_output = os.environ.get("GITHUB_OUTPUT", "")
        if github_output:
            with open(github_output, "a") as f:
                f.write(f"tests-passed={passed}\n")
                f.write(f"tests-failed={failed}\n")
                f.write(f"tests-skipped={skipped}\n")
                f.write(f"test-results-file={results_file}\n")
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
            print(
                "❌ Some tests failed. Check the detailed results for more information."
            )
            return 1

        print("✅ All tests completed successfully!")
        return 0

    except Exception as e:
        print(f"❌ Error parsing test results: {e}")
        return 1


if __name__ == "__main__":
    results_file = sys.argv[1] if len(sys.argv) > 1 else "test_results.json"
    exit_code = parse_test_results(results_file)
    sys.exit(exit_code)

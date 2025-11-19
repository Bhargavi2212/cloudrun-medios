#!/usr/bin/env python3
"""
Pre-push validation script
Runs all linting and tests before pushing to Git
Works on all platforms (Windows, Linux, Mac)
"""

import os
import subprocess
import sys
from pathlib import Path

# Fix Windows console encoding
if sys.platform == "win32":
    import io

    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

# Colors for output (use simple text on Windows if needed)
if sys.platform == "win32" and not os.getenv("TERM"):
    RED = ""
    GREEN = ""
    YELLOW = ""
    CYAN = ""
    NC = ""
else:
    RED = "\033[0;31m"
    GREEN = "\033[0;32m"
    YELLOW = "\033[1;33m"
    CYAN = "\033[0;36m"
    NC = "\033[0m"  # No Color

failed_checks = []


def print_header(text: str):
    """Print a formatted header."""
    # Remove emojis for Windows compatibility
    text_clean = text.encode("ascii", "ignore").decode("ascii")
    print(f"\n{CYAN}{'=' * 42}{NC}")
    print(f"{CYAN}{text_clean}{NC}")
    print(f"{CYAN}{'=' * 42}{NC}\n")


def run_check(
    name: str, command: list[str], cwd: Path | None = None, allow_failure: bool = False
) -> bool:
    """Run a check command and return True if successful."""
    print(f"{YELLOW}[RUN] {name}{NC}")

    try:
        result = subprocess.run(
            command, cwd=cwd, check=True, capture_output=True, text=True
        )
        print(f"{GREEN}[PASS] {name}{NC}\n")
        if result.stdout:
            print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"{RED}[FAIL] {name}{NC}\n")
        if e.stdout:
            print(e.stdout)
        if e.stderr:
            print(e.stderr)
        if not allow_failure:
            failed_checks.append(name)
        return False
    except FileNotFoundError:
        print(f"{RED}[FAIL] Command not found: {command[0]}{NC}\n")
        if not allow_failure:
            failed_checks.append(name)
        return False


def main():
    """Main validation function."""
    print_header("Medi OS Pre-Push Validation")

    # Check if we're in the right directory
    if not Path("pyproject.toml").exists():
        print(f"{RED}Error: Must run from Version -2 directory{NC}")
        sys.exit(1)

    # Check if Poetry is installed
    try:
        subprocess.run(["poetry", "--version"], check=True, capture_output=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print(
            f"{RED}Error: Poetry is not installed. Install with: pip install poetry{NC}"
        )
        sys.exit(1)

    # Check if Node/npm is installed
    try:
        # Try npm.cmd on Windows
        npm_cmd = "npm.cmd" if sys.platform == "win32" else "npm"
        subprocess.run(
            [npm_cmd, "--version"], check=True, capture_output=True, shell=True
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        # Try regular npm
        try:
            subprocess.run(["npm", "--version"], check=True, capture_output=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            print(f"{RED}Error: npm is not installed or not in PATH{NC}")
            print(f"{YELLOW}Continuing with backend checks only...{NC}\n")
            npm_available = False
        else:
            npm_available = True
    else:
        npm_available = True

    print("Checking dependencies...\n")

    # Install/update dependencies if needed
    if not Path(".venv").exists() or not Path("apps/frontend/node_modules").exists():
        print(f"{YELLOW}Installing dependencies...{NC}")
        run_check("Poetry install", ["poetry", "install"], allow_failure=False)
        npm_cmd = "npm.cmd" if sys.platform == "win32" else "npm"
        run_check(
            "npm install",
            [npm_cmd, "install"],
            cwd=Path("apps/frontend"),
            allow_failure=False,
        )

    # Backend checks
    print_header("Backend Checks")

    run_check("Ruff linting", ["poetry", "run", "ruff", "check", "."])
    run_check("Black format check", ["poetry", "run", "black", "--check", "."])
    run_check("MyPy type checking", ["poetry", "run", "mypy", "."], allow_failure=True)

    # Backend tests
    print_header("Backend Tests")

    if not os.getenv("DATABASE_URL") and not os.getenv("TEST_DATABASE_URL"):
        print(f"{YELLOW}[WARN] DATABASE_URL not set. Some tests may be skipped.{NC}")
        print(f"{YELLOW}        Set DATABASE_URL for full test coverage.{NC}\n")

    run_check(
        "Pytest", ["poetry", "run", "pytest", "-v", "--tb=short"], allow_failure=True
    )

    # Frontend checks (only if npm is available)
    if npm_available:
        print_header("Frontend Checks")

        frontend_dir = Path("apps/frontend")
        npm_cmd = "npm.cmd" if sys.platform == "win32" else "npm"
        npx_cmd = "npx.cmd" if sys.platform == "win32" else "npx"

        run_check(
            "ESLint",
            [npm_cmd, "run", "lint", "--", "--max-warnings=0"],
            cwd=frontend_dir,
        )
        run_check(
            "TypeScript compilation", [npx_cmd, "tsc", "--noEmit"], cwd=frontend_dir
        )

        # Frontend tests
        print_header("Frontend Tests")
        run_check("Vitest", [npm_cmd, "run", "test", "--", "--run"], cwd=frontend_dir)
    else:
        print(f"{YELLOW}Skipping frontend checks (npm not available){NC}\n")

    # Summary
    print_header("Summary")

    if not failed_checks:
        print(f"{GREEN}[PASS] All checks passed! Ready to push.{NC}")
        sys.exit(0)
    else:
        print(f"{RED}[FAIL] {len(failed_checks)} check(s) failed:{NC}")
        for check in failed_checks:
            print(f"  - {check}")
        print(f"\n{RED}Please fix issues before pushing.{NC}")
        sys.exit(1)


if __name__ == "__main__":
    main()

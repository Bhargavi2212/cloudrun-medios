#!/usr/bin/env python3
"""
Test runner for Medi OS Kiroween Edition v2.0

This script runs comprehensive tests for database models, connections,
medical safety, and privacy compliance.
"""

import asyncio
import logging
import subprocess
import sys
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def run_command(command: list, description: str) -> bool:
    """
    Run a command and return success status.
    
    Args:
        command: Command to run as list
        description: Description for logging
        
    Returns:
        True if command succeeded, False otherwise
    """
    logger.info(f"🧪 {description}...")
    
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent
        )
        
        if result.returncode == 0:
            logger.info(f"✅ {description} passed")
            if result.stdout.strip():
                print(result.stdout)
            return True
        else:
            logger.error(f"❌ {description} failed")
            if result.stderr.strip():
                print("STDERR:", result.stderr)
            if result.stdout.strip():
                print("STDOUT:", result.stdout)
            return False
            
    except Exception as e:
        logger.error(f"❌ {description} failed with exception: {e}")
        return False


def main() -> int:
    """Main test runner function."""
    logger.info("🏥 Starting Medi OS Test Suite...")
    
    # Check if we're in the right directory
    if not Path("pyproject.toml").exists():
        logger.error("❌ Must run from Kiroween directory (pyproject.toml not found)")
        return 1
        
    # Install dependencies if needed
    logger.info("📦 Installing test dependencies...")
    install_result = subprocess.run([
        sys.executable, "-m", "pip", "install", "-e", ".[dev]"
    ], capture_output=True, text=True)
    
    if install_result.returncode != 0:
        logger.error("❌ Failed to install dependencies")
        print(install_result.stderr)
        return 1
        
    logger.info("✅ Dependencies installed")
    
    # Test categories to run
    test_categories = [
        {
            "name": "Medical Safety Validation",
            "command": [sys.executable, "scripts/validate_medical_safety.py"],
        },
        {
            "name": "Privacy Validation", 
            "command": [sys.executable, "scripts/validate_privacy.py"],
        },
        {
            "name": "PHI Data Check",
            "command": [sys.executable, "scripts/check_no_phi.py"],
        },
        {
            "name": "Database Model Tests",
            "command": [sys.executable, "-m", "pytest", "tests/test_database_models.py", "-v"],
        },
        {
            "name": "Database Connection Tests",
            "command": [sys.executable, "-m", "pytest", "tests/test_database_connection.py", "-v"],
        },
        {
            "name": "All Unit Tests",
            "command": [sys.executable, "-m", "pytest", "tests/", "-v", "--tb=short"],
        },
        {
            "name": "Medical Safety Tests",
            "command": [sys.executable, "-m", "pytest", "-m", "medical_safety", "-v"],
        },
        {
            "name": "Privacy Tests",
            "command": [sys.executable, "-m", "pytest", "-m", "privacy", "-v"],
        },
    ]
    
    # Run all tests
    passed_tests = 0
    total_tests = len(test_categories)
    
    for test_category in test_categories:
        success = run_command(test_category["command"], test_category["name"])
        if success:
            passed_tests += 1
            
    # Summary
    logger.info(f"\n📊 Test Results Summary:")
    logger.info(f"   Tests passed: {passed_tests}/{total_tests}")
    
    if passed_tests == total_tests:
        logger.info("✅ All tests passed! Database implementation is ready.")
        return 0
    else:
        logger.error("❌ Some tests failed. Please fix issues before proceeding.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
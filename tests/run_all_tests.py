"""
Comprehensive Test Runner for EG-QGEM Framework
==============================================

This script runs all tests for the EG-QGEM research system.
"""

import unittest
import sys
import os
from io import StringIO

# Add project to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

# Import test modules
from test_theory import *
from test_simulations import *
from test_experiments import *


class TestResult:
    """Custom test result handler."""

    def __init__(self):
        self.successes = 0
        self.failures = 0
        self.errors = 0
        self.total = 0
        self.details = []

    def add_result(self, result, module_name):
        """Add results from a test module."""
        self.total += result.testsRun
        self.failures += len(result.failures)
        self.errors += len(result.errors)
        self.successes += result.testsRun - len(result.failures) - len(result.errors)

        self.details.append({
            'module': module_name,
            'tests_run': result.testsRun,
            'failures': len(result.failures),
            'errors': len(result.errors),
            'success_rate': (result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun if result.testsRun > 0 else 0
        })

    def print_summary(self):
        """Print comprehensive test summary."""
        print("\n" + "="*60)
        print("EG-QGEM Framework Test Summary")
        print("="*60)

        for detail in self.details:
            print(f"\n{detail['module']}:")
            print(f"  Tests run: {detail['tests_run']}")
            print(f"  Successes: {detail['tests_run'] - detail['failures'] - detail['errors']}")
            print(f"  Failures: {detail['failures']}")
            print(f"  Errors: {detail['errors']}")
            print(f"  Success rate: {detail['success_rate']:.1%}")

        print(f"\nOverall Results:")
        print(f"  Total tests: {self.total}")
        print(f"  Successes: {self.successes}")
        print(f"  Failures: {self.failures}")
        print(f"  Errors: {self.errors}")

        if self.total > 0:
            success_rate = self.successes / self.total
            print(f"  Overall success rate: {success_rate:.1%}")

        print("="*60)

        return self.failures == 0 and self.errors == 0


def run_module_tests(module_name, test_classes):
    """Run tests for a specific module."""
    print(f"\nRunning {module_name} tests...")
    print("-" * 40)

    # Create test suite
    suite = unittest.TestSuite()

    for test_class in test_classes:
        suite.addTest(unittest.makeSuite(test_class))

    # Capture output
    stream = StringIO()
    runner = unittest.TextTestRunner(stream=stream, verbosity=2)
    result = runner.run(suite)

    # Print captured output
    output = stream.getvalue()
    print(output)

    return result


def main():
    """Main test runner."""
    print("Starting EG-QGEM Framework Test Suite")
    print("====================================")

    test_result = TestResult()

    # Define test modules and their test classes
    test_modules = [
        {
            'name': 'Theory Module',
            'classes': [TestConstants, TestEntanglementTensor, TestModifiedEinsteinSolver, TestIntegration]
        },
        {
            'name': 'Simulations Module',
            'classes': [TestSpacetimeEmergenceSimulator, TestBlackHoleSimulator, TestSimulationIntegration]
        },
        {
            'name': 'Experiments Module',
            'classes': [TestExperimentalPredictions, TestPredictionAccuracy]
        }
    ]

    # Run tests for each module
    all_passed = True
    for module in test_modules:
        try:
            result = run_module_tests(module['name'], module['classes'])
            test_result.add_result(result, module['name'])

            if not result.wasSuccessful():
                all_passed = False

        except Exception as e:
            print(f"Error running {module['name']} tests: {e}")
            all_passed = False

    # Print comprehensive summary
    success = test_result.print_summary()

    if success and all_passed:
        print("\n✅ All tests passed successfully!")
        return 0
    else:
        print("\n❌ Some tests failed.")
        return 1


if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)

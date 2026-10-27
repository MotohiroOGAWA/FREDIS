import unittest
import os

if __name__ == "__main__":
    # Discover and run all tests in the "tests" folder
    loader = unittest.TestLoader()
    suite = loader.discover(start_dir="tests", pattern="Test*.py")

    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite)

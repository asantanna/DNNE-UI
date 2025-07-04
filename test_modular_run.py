#!/usr/bin/env python3
"""
Test runner for modular export that sets up proper Python path
"""
import sys
import os
from pathlib import Path

# Add the export directory to Python path
export_dir = Path("export_system/exports/MNIST-Test")
sys.path.insert(0, str(export_dir))

# Now we can import and run
import runner
import asyncio

if __name__ == "__main__":
    asyncio.run(runner.main())
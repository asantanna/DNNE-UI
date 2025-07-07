#!/usr/bin/env python3
"""
Check all required dependencies for DNNE tests.
Exits with error code 1 if any dependencies are missing.
"""

import sys
import os
from pathlib import Path


def check_dependencies():
    """Verify all required dependencies are installed."""
    errors = []
    warnings = []
    
    print("Checking DNNE test dependencies...\n")
    
    # Check PyTorch
    try:
        import torch
        print(f"✓ PyTorch {torch.__version__}")
        
        # Check CUDA availability
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            device_count = torch.cuda.device_count()
            print(f"✓ CUDA available: {device_name} ({device_count} device(s))")
        else:
            print("ℹ CUDA not available - GPU tests will fail")
            warnings.append("CUDA not available")
            
    except ImportError as e:
        errors.append(f"PyTorch not installed: {e}")
    
    # Check torchvision
    try:
        import torchvision
        print(f"✓ torchvision {torchvision.__version__}")
    except ImportError as e:
        errors.append(f"torchvision not installed: {e}")
    
    # Check numpy
    try:
        import numpy as np
        print(f"✓ numpy {np.__version__}")
    except ImportError as e:
        errors.append(f"numpy not installed: {e}")
    
    # Check Isaac Gym
    # Note: Isaac Gym must be imported before torch in actual usage
    # Since we already imported torch above, we need to check differently
    import subprocess
    try:
        # Test Isaac Gym import in a fresh Python process
        result = subprocess.run(
            [sys.executable, "-c", "import isaacgym; print('OK')"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0 and 'OK' in result.stdout:
            print("✓ Isaac Gym available (tested in separate process)")
        else:
            errors.append(f"Isaac Gym not installed or failed to import: {result.stderr}")
    except Exception as e:
        errors.append(f"Failed to test Isaac Gym: {e}")
    
    # Check pytest and test dependencies
    try:
        import pytest
        print(f"✓ pytest {pytest.__version__}")
    except ImportError as e:
        errors.append(f"pytest not installed: {e}")
    
    try:
        import pytest_timeout
        print("✓ pytest-timeout available")
    except ImportError:
        warnings.append("pytest-timeout not installed - timeout protection unavailable")
    
    try:
        import pytest_asyncio
        print("✓ pytest-asyncio available")
    except ImportError:
        errors.append("pytest-asyncio not installed - async tests will fail")
    
    # Check MNIST data location
    print("\nChecking data availability...")
    data_path = os.environ.get('DNNE_TEST_DATA_PATH', './data')
    mnist_path = Path(data_path) / 'MNIST' / 'raw'
    
    print(f"Data path: {data_path}")
    
    if mnist_path.exists():
        files = list(mnist_path.glob('*'))
        gz_files = [f for f in files if f.suffix == '.gz']
        
        if len(gz_files) >= 4:  # Should have at least 4 .gz files
            print(f"✓ MNIST data available at {mnist_path}")
            print(f"  Found {len(gz_files)} compressed files")
        else:
            print(f"⚠ MNIST data incomplete at {mnist_path}")
            print(f"  Found only {len(gz_files)} files (expected 4+)")
            warnings.append(f"MNIST data incomplete at {mnist_path}")
    else:
        download = os.environ.get('DNNE_TEST_DOWNLOAD', 'true').lower() == 'true'
        if download:
            print(f"ℹ MNIST data will be downloaded to {data_path}")
        else:
            print(f"⚠ MNIST data not found at {mnist_path} and download=false")
            warnings.append(f"MNIST data not found and download disabled")
    
    # Check DNNE project structure
    print("\nChecking DNNE project structure...")
    project_root = Path(__file__).parent.parent
    
    required_dirs = [
        'custom_nodes',
        'custom_nodes/ml_nodes',
        'custom_nodes/robotics_nodes',
        'export_system',
        'export_system/node_exporters',
        'export_system/templates',
    ]
    
    for dir_name in required_dirs:
        dir_path = project_root / dir_name
        if dir_path.exists():
            print(f"✓ {dir_name}")
        else:
            errors.append(f"Required directory not found: {dir_name}")
    
    # Summary
    print("\n" + "="*60)
    
    if warnings:
        print("\n⚠ Warnings:")
        for warning in warnings:
            print(f"  - {warning}")
    
    if errors:
        print("\n❌ Missing dependencies:")
        for error in errors:
            print(f"  - {error}")
        print("\nPlease install missing dependencies before running tests.")
        sys.exit(1)
    else:
        print("\n✅ All required dependencies are satisfied!")
        if warnings:
            print("   (Some optional features may be limited)")
        return 0


if __name__ == "__main__":
    check_dependencies()
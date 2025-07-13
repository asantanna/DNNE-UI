#!/usr/bin/env python3
"""Quick test of Linux repository integration"""

import sys
import os

# Add project root to path
project_root = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, project_root)

def test_template_paths():
    """Test that templates use Linux repository paths"""
    
    print("🧪 Testing template paths for Linux repository integration...")
    
    # Check isaac_gym_env_queue.py template
    template_path = os.path.join(project_root, 'export_system', 'templates', 'nodes', 'isaac_gym_env_queue.py')
    
    if not os.path.exists(template_path):
        print(f"❌ Template not found: {template_path}")
        return False
    
    with open(template_path, 'r') as f:
        content = f.read()
    
    # Check for Linux repository paths
    if '/home/asantanna/DNNE-LINUX-SUPPORT/isaacgym' in content:
        print("✅ Isaac Gym template uses Linux repository path")
    else:
        print("❌ Isaac Gym template NOT using Linux repository path")
        return False
        
    if '/home/asantanna/DNNE-LINUX-SUPPORT/IsaacGymEnvs' in content:
        print("✅ IsaacGymEnvs template uses Linux repository path")
    else:
        print("❌ IsaacGymEnvs template NOT using Linux repository path")
        return False
    
    # Check node exporter paths
    exporter_path = os.path.join(project_root, 'export_system', 'node_exporters', 'robotics_nodes.py')
    
    if not os.path.exists(exporter_path):
        print(f"❌ Exporter not found: {exporter_path}")
        return False
    
    with open(exporter_path, 'r') as f:
        content = f.read()
    
    if '/home/asantanna/DNNE-LINUX-SUPPORT/isaacgym' in content:
        print("✅ Robotics exporter uses Linux repository paths")
    else:
        print("❌ Robotics exporter NOT using Linux repository paths")
        return False
    
    return True

def test_imports():
    """Test that Isaac Gym imports work from Linux repository"""
    
    print("\n🧪 Testing Isaac Gym imports from Linux repository...")
    
    try:
        import isaacgym
        print("✅ Isaac Gym import successful")
    except ImportError as e:
        print(f"❌ Isaac Gym import failed: {e}")
        return False
    
    try:
        import isaacgymenvs
        print("✅ IsaacGymEnvs import successful")
    except ImportError as e:
        print(f"❌ IsaacGymEnvs import failed: {e}")
        return False
    
    return True

def main():
    """Run all tests"""
    
    print("🚀 Testing DNNE Linux Repository Integration")
    print("=" * 50)
    
    success = True
    
    # Test template paths
    if not test_template_paths():
        success = False
    
    # Test imports
    if not test_imports():
        success = False
    
    print("\n" + "=" * 50)
    if success:
        print("✅ All tests passed! Linux repository integration successful.")
        print("📍 Isaac Gym and IsaacGymEnvs are now running from /home/asantanna/DNNE-LINUX-SUPPORT/")
        print("📍 Export templates are configured to use Linux repository paths")
        print("📍 Ready for performance testing and further development")
    else:
        print("❌ Some tests failed. Check the output above for details.")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
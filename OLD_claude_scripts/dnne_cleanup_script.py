#!/usr/bin/env python3
"""
DNNE-UI Backend Cleanup Script
Safely disables SD-specific components by renaming them.
"""

import os
import shutil
from pathlib import Path
import argparse

class ComfyUICleanup:
    def __init__(self, comfy_root: str):
        self.root = Path(comfy_root)
        self.disabled_prefix = "_DISABLED_"
        
    def safe_rename(self, path: Path, action: str = "disable"):
        """Safely rename files/directories"""
        if not path.exists():
            print(f"  Skip: {path} (doesn't exist)")
            return False
            
        if action == "disable":
            new_name = path.parent / (self.disabled_prefix + path.name)
            if new_name.exists():
                print(f"  Skip: {path} (already disabled)")
                return False
            path.rename(new_name)
            print(f"  ✓ Disabled: {path.name} → {new_name.name}")
            
        elif action == "enable":
            if not path.name.startswith(self.disabled_prefix):
                print(f"  Skip: {path} (not disabled)")
                return False
            new_name = path.parent / path.name[len(self.disabled_prefix):]
            path.rename(new_name)
            print(f"  ✓ Enabled: {path.name} → {new_name.name}")
            
        return True
    
    def disable_sd_components(self):
        """Disable all SD-specific components"""
        print("🧹 Disabling SD components...")
        
        # SD model directories
        models_dir = self.root / "models"
        if models_dir.exists():
            print("\n📁 Model directories:")
            sd_model_dirs = [
                "checkpoints", "vae", "loras", "embeddings", 
                "controlnet", "clip_vision", "diffusion_models",
                "upscale_models", "hypernetworks"
            ]
            for dirname in sd_model_dirs:
                self.safe_rename(models_dir / dirname)
        
        # SD components in comfy/
        comfy_dir = self.root / "comfy"
        if comfy_dir.exists():
            print("\n🔧 Comfy components:")
            sd_comfy_dirs = [
                "samplers", "sd1", "sd2", "sdxl", "sd3", 
                "ldm", "cldm", "t2i_adapter", "diffusers_convert",
                "clip", "controlnet", "model_detection",
                "sample", "k_diffusion", "lora", "model_base",
                "model_patcher", "diffusers_load", "extras"
            ]
            for dirname in sd_comfy_dirs:
                self.safe_rename(comfy_dir / dirname)
        
        # Other SD directories
        print("\n📦 Other SD directories:")
        other_sd_dirs = ["comfy_extras", "notebooks", "input", "output", "temp"]
        for dirname in other_sd_dirs:
            self.safe_rename(self.root / dirname)
            
        print("\n✅ SD components disabled!")
        
    def enable_sd_components(self):
        """Re-enable SD components"""
        print("🔄 Re-enabling SD components...")
        
        # Find all disabled items
        disabled_items = []
        for item in self.root.rglob(f"{self.disabled_prefix}*"):
            disabled_items.append(item)
            
        print(f"\n📋 Found {len(disabled_items)} disabled items:")
        for item in disabled_items:
            self.safe_rename(item, "enable")
            
        print("\n✅ SD components re-enabled!")
    
    def create_robotics_structure(self):
        """Create new robotics-focused directory structure"""
        print("🤖 Creating robotics structure...")
        
        new_dirs = [
            "robotics_nodes",
            "export",
            "export/templates", 
            "examples",
            "models/neural_networks",
            "models/configurations"
        ]
        
        for dirname in new_dirs:
            dir_path = self.root / dirname
            dir_path.mkdir(parents=True, exist_ok=True)
            print(f"  ✓ Created: {dirname}")
            
        # Create __init__.py files
        init_dirs = ["robotics_nodes", "export"]
        for dirname in init_dirs:
            init_file = self.root / dirname / "__init__.py"
            init_file.touch()
            print(f"  ✓ Created: {dirname}/__init__.py")
            
        print("\n✅ Robotics structure created!")
    
    def update_requirements(self):
        """Update requirements.txt for robotics"""
        print("📋 Updating requirements.txt...")
        
        # Backup original
        req_file = self.root / "requirements.txt"
        if req_file.exists():
            backup_file = self.root / "_DISABLED_requirements.txt"
            shutil.copy2(req_file, backup_file)
            print(f"  ✓ Backed up: requirements.txt → {backup_file.name}")
        
        # Create minimal robotics requirements
        minimal_requirements = """torch>=2.0.0
torchvision  
numpy
Pillow
psutil
tqdm
safetensors
websockets
aiohttp
aiofiles
spandrel
"""
        
        with open(req_file, 'w') as f:
            f.write(minimal_requirements)
        print("  ✓ Created minimal requirements.txt")
        
    def status(self):
        """Show current status"""
        print("📊 DNNE-UI Backend Status:")
        
        disabled_count = len(list(self.root.rglob(f"{self.disabled_prefix}*")))
        print(f"  Disabled components: {disabled_count}")
        
        robotics_dirs = ["robotics_nodes", "export", "examples"]
        existing_robotics = sum(1 for d in robotics_dirs if (self.root / d).exists())
        print(f"  Robotics directories: {existing_robotics}/{len(robotics_dirs)}")
        
        # Check for key files
        key_files = ["main.py", "server.py", "execution.py", "nodes.py"]
        existing_files = sum(1 for f in key_files if (self.root / f).exists())
        print(f"  Core files: {existing_files}/{len(key_files)}")

def main():
    parser = argparse.ArgumentParser(description="DNNE-UI Backend Cleanup Tool")
    parser.add_argument("action", choices=["disable", "enable", "setup", "status"], 
                       help="Action to perform")
    parser.add_argument("--root", default=".", 
                       help="Path to ComfyUI root directory")
    
    args = parser.parse_args()
    
    cleanup = ComfyUICleanup(args.root)
    
    if args.action == "disable":
        cleanup.disable_sd_components()
    elif args.action == "enable":
        cleanup.enable_sd_components()
    elif args.action == "setup":
        cleanup.disable_sd_components()
        cleanup.create_robotics_structure()
        cleanup.update_requirements()
        print("\n🎉 DNNE-UI backend setup complete!")
    elif args.action == "status":
        cleanup.status()

if __name__ == "__main__":
    main()
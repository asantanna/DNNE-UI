#!/usr/bin/env python3
"""Debug why episodes are not completing in DNNE"""

import subprocess
import sys
import os

def add_progress_debug():
    """Add debug logging to track progress buffer"""
    # Add to base_environment.py step function
    base_env_file = "export_system/exports/Cartpole_PPO/environments/base_environment.py"
    
    with open(base_env_file, 'r') as f:
        content = f.read()
    
    # Add debug logging after progress increment
    debug_code = '''
        # Debug progress tracking
        if self.step_count % 100 == 0:  # Every 100 steps
            max_progress = self.progress_buf.max().item()
            min_progress = self.progress_buf.min().item()
            print(f"[Episode Debug] Step {self.step_count}: progress_buf range [{min_progress}, {max_progress}], max_episode_length={getattr(self, 'max_episode_length', 'not set')}")
'''
    
    if "[Episode Debug]" not in content:
        content = content.replace('# Update progress\n        self.progress_buf += 1\n        self.step_count += 1',
                                '# Update progress\n        self.progress_buf += 1\n        self.step_count += 1' + debug_code)
        
        with open(base_env_file, 'w') as f:
            f.write(content)
    
    # Also add debug to check_termination in cartpole
    cartpole_file = "export_system/exports/Cartpole_PPO/environments/cartpole_environment.py"
    
    with open(cartpole_file, 'r') as f:
        content = f.read()
    
    if "[Episode Debug] Termination" not in content:
        debug_code = '''
        # Debug termination conditions
        if self.step_count % 100 == 0:
            cart_pos_max = torch.abs(cart_pos).max().item()
            pole_angle_max = torch.abs(pole_angle).max().item()
            progress_max = self.progress_buf.max().item()
            reset_count = reset.sum().item()
            print(f"[Episode Debug] Termination check: max_cart_pos={cart_pos_max:.3f} (limit={self.reset_dist}), max_pole_angle={pole_angle_max:.3f} (limit={np.pi/2:.3f}), max_progress={progress_max} (limit={self.max_episode_length}), resets={reset_count}")
'''
        content = content.replace('# Episode timeout\n        reset = torch.where(self.progress_buf >= self.max_episode_length - 1, torch.ones_like(reset), reset)',
                                '# Episode timeout\n        reset = torch.where(self.progress_buf >= self.max_episode_length - 1, torch.ones_like(reset), reset)' + debug_code)
        
        with open(cartpole_file, 'w') as f:
            f.write(content)
    
    # Add initial state debug
    isaac_env_file = "export_system/exports/Cartpole_PPO/nodes/isaacgymenvnode_7.py"
    
    with open(isaac_env_file, 'r') as f:
        content = f.read()
    
    if "[Episode Debug] Initial" not in content:
        debug_code = '''
            # Debug initial state
            print(f"[Episode Debug] Initial environment state created")
            if hasattr(self.environment, 'max_episode_length'):
                print(f"[Episode Debug] max_episode_length = {self.environment.max_episode_length}")
            if hasattr(self.environment, 'progress_buf'):
                print(f"[Episode Debug] Initial progress_buf shape = {self.environment.progress_buf.shape}")
'''
        content = content.replace('# Reset all environments initially\n            self.environment.reset_environments(torch.arange(self.num_envs))',
                                '# Reset all environments initially\n            self.environment.reset_environments(torch.arange(self.num_envs))' + debug_code)
        
        with open(isaac_env_file, 'w') as f:
            f.write(content)

def run_debug():
    """Run DNNE with episode debug logging"""
    print("Running DNNE with episode completion debugging...")
    
    # First export
    subprocess.run([sys.executable, "claude_scripts/programmatic_export.py", "Cartpole_PPO"], 
                   check=True, capture_output=True)
    
    # Add debug logging
    add_progress_debug()
    
    # Run with short timeout
    cmd = [
        sys.executable,
        "export_system/exports/Cartpole_PPO/runner.py",
        "--epochs=10",
        "--timeout=20s",
        "--headless"
    ]
    
    print(f"Command: {' '.join(cmd)}")
    print("="*80)
    
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                             universal_newlines=True, bufsize=1)
    
    episode_count = 0
    max_progress_seen = 0
    termination_checks = []
    
    for line in iter(process.stdout.readline, ''):
        if line:
            if "[Episode Debug]" in line:
                print(line.rstrip())
                
                # Track max progress
                if "progress_buf range" in line:
                    try:
                        parts = line.split("range [")[1].split("]")[0].split(", ")
                        max_val = int(float(parts[1]))
                        max_progress_seen = max(max_progress_seen, max_val)
                    except:
                        pass
                
                # Track termination checks
                if "Termination check:" in line:
                    termination_checks.append(line.rstrip())
            
            elif "Episode" in line and "episode return" in line:
                episode_count += 1
                print(f"✅ {line.rstrip()}")
    
    process.wait()
    
    print("\n" + "="*80)
    print(f"Episodes completed: {episode_count}")
    print(f"Max progress seen: {max_progress_seen}")
    print(f"Termination checks: {len(termination_checks)}")
    
    if episode_count == 0:
        print("\n❌ No episodes completed!")
        print("Possible issues:")
        print(f"1. Progress buffer not reaching max_episode_length (500)")
        print(f"2. Termination conditions not triggering")
        print(f"3. Reset not being called when termination occurs")
        
        if termination_checks:
            print("\nLast termination check:")
            print(termination_checks[-1])

if __name__ == "__main__":
    # Activate conda environment
    activate_cmd = "source /home/asantanna/miniconda/bin/activate DNNE_PY38"
    subprocess.run(activate_cmd, shell=True, executable="/bin/bash")
    
    os.chdir("/mnt/e/ALS-Projects/DNNE/DNNE-UI")
    run_debug()
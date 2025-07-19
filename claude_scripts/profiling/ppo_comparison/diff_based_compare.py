#!/usr/bin/env python3
"""
PPO log comparison tool using diff for proper alignment
"""
import re
import sys
import subprocess
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import argparse

# Add parent directory to Python path to import dnne_config
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
from dnne_config import config

class DiffBasedComparer:
    def __init__(self, width: int = 80, ignore_shared_differences: bool = False):
        self.width = width
        self.debug = False
        self.ignore_shared_differences = ignore_shared_differences
        
    def preprocess_line(self, line: str) -> str:
        """Preprocess a log line to remove changing values for structural comparison"""
        # Remove the shared attribute from DNNE_DEBUG lines
        if '[DNNE_DEBUG]' in line:
            line = re.sub(r'\[DNNE_DEBUG\] [DIB]/', '[DNNE_DEBUG] ?/', line)
        
        # Normalize caller information
        line = re.sub(r'called by \S+', 'called by CALLER', line)
        line = re.sub(r'call #\d+ by \S+', 'call #<NUM> by CALLER', line)
        
        # Remove timestamps
        line = re.sub(r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}', 'TIMESTAMP', line)
        
        # Remove specific numeric values but keep structure
        line = re.sub(r'-?\d+\.\d+[eE]?[-+]?\d*', '<NUM>', line)  # Floats
        line = re.sub(r'\b\d{2,}\b', '<NUM>', line)  # Large integers
        
        # Normalize step numbers and cycles
        line = re.sub(r'Step \d+:', 'Step <NUM>:', line)
        line = re.sub(r'CYCLE \d+', 'CYCLE <NUM>', line)
        line = re.sub(r'call #\d+', 'call #<NUM>', line)
        
        # Normalize file paths (but NOT category names like /PPO_CYCLE)
        line = re.sub(r'(/home/[^\s]+|/mnt/[^\s]+|/usr/[^\s]+|/tmp/[^\s]+)', '/PATH', line)
        line = re.sub(r'(/[a-zA-Z0-9_\-\.]+){3,}', '/PATH', line)
        
        # Normalize tensor shapes
        line = re.sub(r'torch\.Size\(\[[^\]]+\]\)', 'torch.Size([<NUM>])', line)
        
        # Normalize arrays
        line = re.sub(r'\[[-\d., ]+\]', '[<NUM>]', line)
        
        # Normalize hash values
        line = re.sub(r'hash: -?\d+', 'hash: <NUM>', line)
        
        return line
    
    def only_shared_attr_differs(self, line1: str, line2: str) -> bool:
        """Check if two lines differ only in the shared attribute (D/I/B)"""
        # Check if both lines have DNNE_DEBUG
        if '[DNNE_DEBUG]' not in line1 or '[DNNE_DEBUG]' not in line2:
            return False
        
        # Replace shared attributes with same placeholder and compare
        normalized1 = re.sub(r'\[DNNE_DEBUG\] [DIB]/', '[DNNE_DEBUG] ?/', line1)
        normalized2 = re.sub(r'\[DNNE_DEBUG\] [DIB]/', '[DNNE_DEBUG] ?/', line2)
        
        return normalized1 == normalized2
    
    def save_preprocessed(self, filepath: Path) -> Path:
        """Create preprocessed version of a file"""
        output_path = config.get_temp_dir() / f"preproc_{filepath.name}"
        
        with open(filepath, 'r') as f_in, open(output_path, 'w') as f_out:
            for line in f_in:
                preprocessed = self.preprocess_line(line.rstrip())
                f_out.write(preprocessed + '\n')
        
        print(f"Saved preprocessed: {output_path}")
        return output_path
    
    def run_diff(self, file1: Path, file2: Path) -> List[str]:
        """Run diff -U 0 on two files"""
        try:
            result = subprocess.run(
                ['diff', '-U', '0', str(file1), str(file2)],
                capture_output=True,
                text=True
            )
            return result.stdout.splitlines()
        except Exception as e:
            print(f"Error running diff: {e}")
            return []
    
    def parse_unified_diff(self, diff_output: List[str], 
                          orig1_lines: List[str], orig2_lines: List[str],
                          prep1_lines: List[str], prep2_lines: List[str]) -> List[Tuple[str, Optional[int], Optional[int]]]:
        """
        Parse unified diff output to create alignment.
        Returns list of (type, index1, index2) where type is:
        - 'match': lines match exactly (white)
        - 'yellow': lines match only after preprocessing
        - 'hunk': part of a diff hunk (red/green)
        """
        alignment = []
        i1, i2 = 0, 0  # Current position in files
        debug = self.debug  # Use the instance debug flag
        
        i = 0
        while i < len(diff_output):
            line = diff_output[i]
            
            if line.startswith('@@'):
                # Parse hunk header: @@ -start1,count1 +start2,count2 @@
                match = re.match(r'@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@', line)
                if not match:
                    i += 1
                    continue
                    
                start1 = int(match.group(1)) - 1  # Convert to 0-based
                count1 = int(match.group(2)) if match.group(2) else 1
                start2 = int(match.group(3)) - 1
                count2 = int(match.group(4)) if match.group(4) else 1
                
                if debug:
                    print(f"\nDEBUG: Processing hunk {line}")
                    print(f"  Current positions: i1={i1}, i2={i2}")
                    print(f"  Hunk targets: start1={start1} (count={count1}), start2={start2} (count={count2})")
                
                # Handle lines between hunks
                # Key insight: The diff tells us which lines in the preprocessed files correspond
                # For a zero-length hunk, the position indicates where content would be inserted
                
                if debug:
                    print(f"  Catch-up phase from ({i1},{i2}) to ({start1},{start2})")

                # ALS: special case for empty hunks
                if count1 == 0:
                    start1 += 1
                if count2 == 0:
                    start2 += 1
                    
                # Pair lines between current position and hunk start
                # We need to handle the fact that positions may be offset due to previous edits
                while i1 < start1 or i2 < start2:
                    if i1 < start1 and i2 < start2:
                        # Both files have lines before the hunk
                        if debug:
                            print(f"    Pairing ({i1},{i2}): '{orig1_lines[i1][:30]}...' <-> '{orig2_lines[i2][:30]}...'")
                        
                        if orig1_lines[i1] == orig2_lines[i2]:
                            alignment.append(('match', i1, i2))
                        elif self.ignore_shared_differences and self.only_shared_attr_differs(orig1_lines[i1], orig2_lines[i2]):
                            alignment.append(('match', i1, i2))  # Treat as match if only shared attr differs
                        else:
                            alignment.append(('yellow', i1, i2))
                        i1 += 1
                        i2 += 1
                    elif i1 < start1:
                        # Only file1 has lines left before hunk
                        if debug:
                            print(f"    Red {i1}: '{orig1_lines[i1][:30]}...'")
                        alignment.append(('hunk', i1, None))
                        i1 += 1
                    elif i2 < start2:
                        # Only file2 has lines left before hunk
                        if debug:
                            print(f"    Green {i2}: '{orig2_lines[i2][:30]}...'")
                        alignment.append(('hunk', None, i2))
                        i2 += 1
                
                # Process the hunk itself
                # Read all lines from the hunk
                i += 1
                del_lines = []  # Line indices for deletions
                add_lines = []  # Line indices for additions
                
                # Collect deletion indices
                hunk_i1 = i1
                while i < len(diff_output) and diff_output[i].startswith('-'):
                    del_lines.append(hunk_i1)
                    hunk_i1 += 1
                    i += 1
                
                # Collect addition indices  
                hunk_i2 = i2
                while i < len(diff_output) and diff_output[i].startswith('+'):
                    add_lines.append(hunk_i2)
                    hunk_i2 += 1
                    i += 1
                
                # Create alignment for the hunk using max length
                max_count = max(len(del_lines), len(add_lines))
                for j in range(max_count):
                    left_idx = del_lines[j] if j < len(del_lines) else None
                    right_idx = add_lines[j] if j < len(add_lines) else None
                    alignment.append(('hunk', left_idx, right_idx))
                
                # Update positions after processing the hunk
                i1 = hunk_i1
                i2 = hunk_i2
                
                continue
                
            elif line.startswith('---') or line.startswith('+++'):
                # Skip file headers
                i += 1
                continue
            else:
                i += 1
        
        # Handle any remaining lines after all hunks
        while i1 < len(orig1_lines) and i2 < len(orig2_lines):
            if orig1_lines[i1] == orig2_lines[i2]:
                alignment.append(('match', i1, i2))  # White
            elif self.ignore_shared_differences and self.only_shared_attr_differs(orig1_lines[i1], orig2_lines[i2]):
                alignment.append(('match', i1, i2))  # Treat as match if only shared attr differs
            else:
                alignment.append(('yellow', i1, i2))  # Yellow
            i1 += 1
            i2 += 1
        
        # Handle any trailing lines
        while i1 < len(orig1_lines):
            alignment.append(('hunk', i1, None))
            i1 += 1
            
        while i2 < len(orig2_lines):
            alignment.append(('hunk', None, i2))
            i2 += 1
        
        return alignment
    
    def display_alignment(self, alignment: List[Tuple[str, Optional[int], Optional[int]]],
                         orig1_lines: List[str], orig2_lines: List[str],
                         prep1_lines: List[str], prep2_lines: List[str],
                         name1: str, name2: str):
        """Display the aligned comparison with colors"""
        divider = " | "
        
        # Header
        header1 = name1[:self.width].center(self.width)
        header2 = name2[:self.width].center(self.width)
        print(f"\n{header1}{divider}{header2}")
        print("=" * (self.width * 2 + len(divider)))
        
        # Statistics
        stats = {'match': 0, 'yellow': 0, 'red': 0, 'green': 0, 'both': 0}
        
        # Display each aligned pair
        for align_type, idx1, idx2 in alignment:
            # Get the actual lines (or empty for fillers)
            line1 = orig1_lines[idx1] if idx1 is not None else ""
            line2 = orig2_lines[idx2] if idx2 is not None else ""
            
            # Format for display
            formatted1 = line1[:self.width].ljust(self.width)
            formatted2 = line2[:self.width].ljust(self.width)
            
            if align_type == 'match':
                # White - exact match
                print(f"{formatted1}{divider}{formatted2}")
                stats['match'] += 1
                
            elif align_type == 'yellow':
                # Yellow - matches only after preprocessing
                print(f"\033[33m{formatted1}\033[0m{divider}\033[33m{formatted2}\033[0m")
                stats['yellow'] += 1
                
            elif align_type == 'hunk':
                # Part of a diff hunk - red/green/blank
                if idx1 is not None and idx2 is not None:
                    # Both sides have content
                    print(f"\033[31m{formatted1}\033[0m{divider}\033[32m{formatted2}\033[0m")
                    stats['red'] += 1
                    stats['green'] += 1
                    stats['both'] += 1
                elif idx1 is not None:
                    # Only left side (red)
                    print(f"\033[31m{formatted1}\033[0m{divider}{' ' * self.width}")
                    stats['red'] += 1
                else:
                    # Only right side (green)
                    print(f"{' ' * self.width}{divider}\033[32m{formatted2}\033[0m")
                    stats['green'] += 1
        
        # Summary
        print("\n" + "=" * (self.width * 2 + len(divider)))
        print(f"\nSummary:")
        print(f"  Total lines: {len(alignment)}")
        print(f"  White (exact match): {stats['match']}")
        print(f"  Yellow (match after preprocessing): {stats['yellow']}")
        print(f"  Red (deleted lines): {stats['red']}")
        print(f"  Green (added lines): {stats['green']}")
        print(f"  Both red/green on same line: {stats['both']}")
    
    def compare_files(self, file1: Path, file2: Path):
        """Main comparison function"""
        print(f"Comparing {file1.name} vs {file2.name}")
        
        # Read original files
        with open(file1, 'r') as f:
            orig1_lines = [line.rstrip() for line in f]
        with open(file2, 'r') as f:
            orig2_lines = [line.rstrip() for line in f]
        
        # Create preprocessed files
        prep1_path = self.save_preprocessed(file1)
        prep2_path = self.save_preprocessed(file2)
        
        # Read preprocessed lines for comparison
        with open(prep1_path, 'r') as f:
            prep1_lines = [line.rstrip() for line in f]
        with open(prep2_path, 'r') as f:
            prep2_lines = [line.rstrip() for line in f]
        
        # Run diff
        print(f"Running diff -U 0...")
        diff_output = self.run_diff(prep1_path, prep2_path)
        
        # Parse diff output to get alignment
        alignment = self.parse_unified_diff(diff_output, orig1_lines, orig2_lines, 
                                          prep1_lines, prep2_lines)
        
        # Display the comparison
        self.display_alignment(alignment, orig1_lines, orig2_lines, 
                             prep1_lines, prep2_lines, file1.name, file2.name)

def main():
    parser = argparse.ArgumentParser(description='Compare PPO logs using diff for alignment')
    parser.add_argument('file1', nargs='?', default=str(config.get_temp_dir() / 'dnne_1cycle_final.log'), help=f'First log file (default: {config.get_temp_dir()}/dnne_1cycle_final.log)')
    parser.add_argument('file2', nargs='?', default=str(config.get_temp_dir() / 'ige_1cycle_final.log'), help=f'Second log file (default: {config.get_temp_dir()}/ige_1cycle_final.log)')
    parser.add_argument('--width', '-w', type=int, default=80, help='Column width (default: 80)')
    parser.add_argument('--debug', '-d', action='store_true', help='Enable debug output')
    parser.add_argument('--ignore-shared-differences', action='store_true',
                        help='Ignore differences in shared attributes (D/I/B) when comparing lines')
    
    args = parser.parse_args()
    
    file1 = Path(args.file1)
    file2 = Path(args.file2)
    
    if not file1.exists():
        print(f"Error: {file1} does not exist")
        sys.exit(1)
    if not file2.exists():
        print(f"Error: {file2} does not exist")
        sys.exit(1)
    
    comparer = DiffBasedComparer(width=args.width, ignore_shared_differences=args.ignore_shared_differences)
    comparer.debug = args.debug  # Enable debug if requested
    comparer.compare_files(file1, file2)

if __name__ == "__main__":
    main()
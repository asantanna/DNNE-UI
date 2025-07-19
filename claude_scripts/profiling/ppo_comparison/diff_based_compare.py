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

class DiffBasedComparer:
    def __init__(self, width: int = 80):
        self.width = width
        
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
    
    def save_preprocessed(self, filepath: Path) -> Path:
        """Create preprocessed version of a file"""
        output_path = Path(f"/tmp/preproc_{filepath.name}")
        
        with open(filepath, 'r') as f_in, open(output_path, 'w') as f_out:
            for line in f_in:
                preprocessed = self.preprocess_line(line.rstrip())
                f_out.write(preprocessed + '\n')
        
        print(f"Saved preprocessed: {output_path}")
        return output_path
    
    def run_diff(self, file1: Path, file2: Path) -> List[str]:
        """Run diff --minimal -U 0 on two files"""
        try:
            result = subprocess.run(
                ['diff', '--minimal', '-U', '0', str(file1), str(file2)],
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
        - 'match': lines match structurally
        - 'both': lines are paired by diff but different
        - 'only1': line only in file1
        - 'only2': line only in file2
        """
        alignment = []
        i1, i2 = 0, 0  # Current position in files
        
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
                
                # Add any skipped matching lines before this hunk
                while i1 < start1 and i2 < start2:
                    # These lines exist in both files and match
                    alignment.append(('match', i1, i2))
                    i1 += 1
                    i2 += 1
                
                # Process the hunk content
                i += 1
                hunk_lines = []
                while i < len(diff_output) and not diff_output[i].startswith('@@') and not diff_output[i].startswith('---') and not diff_output[i].startswith('+++'):
                    hunk_lines.append(diff_output[i])
                    i += 1
                
                # Process each line in the hunk
                for hunk_line in hunk_lines:
                    if hunk_line.startswith('-'):
                        # Line only in file1
                        alignment.append(('only1', i1, None))
                        i1 += 1
                    elif hunk_line.startswith('+'):
                        # Line only in file2
                        alignment.append(('only2', None, i2))
                        i2 += 1
                    elif hunk_line.startswith(' '):
                        # Line in both files (context line with -U 0 shouldn't happen often)
                        # Check if they actually match after preprocessing
                        if i1 < len(prep1_lines) and i2 < len(prep2_lines):
                            if prep1_lines[i1] == prep2_lines[i2]:
                                alignment.append(('match', i1, i2))
                            else:
                                alignment.append(('both', i1, i2))
                        i1 += 1
                        i2 += 1
                
                continue
                
            elif line.startswith('---') or line.startswith('+++'):
                # Skip file headers
                i += 1
                continue
            else:
                i += 1
        
        # Add any remaining matching lines
        while i1 < len(orig1_lines) and i2 < len(orig2_lines):
            if i1 < len(prep1_lines) and i2 < len(prep2_lines) and prep1_lines[i1] == prep2_lines[i2]:
                alignment.append(('match', i1, i2))
            else:
                alignment.append(('both', i1, i2))
            i1 += 1
            i2 += 1
        
        # Add any remaining lines from either file
        while i1 < len(orig1_lines):
            alignment.append(('only1', i1, None))
            i1 += 1
            
        while i2 < len(orig2_lines):
            alignment.append(('only2', None, i2))
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
        stats = {'match': 0, 'both': 0, 'only1': 0, 'only2': 0}
        
        # Display each aligned pair
        for align_type, idx1, idx2 in alignment:
            stats[align_type] += 1
            
            # Get the actual lines (or empty for fillers)
            line1 = orig1_lines[idx1] if idx1 is not None else ""
            line2 = orig2_lines[idx2] if idx2 is not None else ""
            
            # Format for display
            formatted1 = line1[:self.width].ljust(self.width)
            formatted2 = line2[:self.width].ljust(self.width)
            
            if align_type == 'match':
                # White - structural match
                print(f"{formatted1}{divider}{formatted2}")
                
            elif align_type == 'both':
                # Yellow - both present but different
                # Double-check they're actually different after preprocessing
                if idx1 is not None and idx2 is not None and idx1 < len(prep1_lines) and idx2 < len(prep2_lines):
                    if prep1_lines[idx1] == prep2_lines[idx2]:
                        # Actually matches, show as white
                        print(f"{formatted1}{divider}{formatted2}")
                        stats['both'] -= 1
                        stats['match'] += 1
                    else:
                        # Truly different, show as yellow
                        print(f"\033[33m{formatted1}\033[0m{divider}\033[33m{formatted2}\033[0m")
                        
            elif align_type == 'only1':
                # Red - only in file1
                print(f"\033[31m{formatted1}\033[0m{divider}{' ' * self.width}")
                
            elif align_type == 'only2':
                # Green - only in file2
                print(f"{' ' * self.width}{divider}\033[32m{formatted2}\033[0m")
        
        # Summary
        print("\n" + "=" * (self.width * 2 + len(divider)))
        print(f"\nSummary:")
        print(f"  Total lines: {len(alignment)}")
        print(f"  White (structural match): {stats['match']}")
        print(f"  Yellow (both but different): {stats['both']}")
        print(f"  Red (only in {name1}): {stats['only1']}")
        print(f"  Green (only in {name2}): {stats['only2']}")
    
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
        print(f"Running diff --minimal -U 0...")
        diff_output = self.run_diff(prep1_path, prep2_path)
        
        # Parse diff output to get alignment
        alignment = self.parse_unified_diff(diff_output, orig1_lines, orig2_lines, 
                                          prep1_lines, prep2_lines)
        
        # Display the comparison
        self.display_alignment(alignment, orig1_lines, orig2_lines, 
                             prep1_lines, prep2_lines, file1.name, file2.name)

def main():
    parser = argparse.ArgumentParser(description='Compare PPO logs using diff for alignment')
    parser.add_argument('file1', help='First log file')
    parser.add_argument('file2', help='Second log file')
    parser.add_argument('--width', '-w', type=int, default=80, help='Column width (default: 80)')
    
    args = parser.parse_args()
    
    file1 = Path(args.file1)
    file2 = Path(args.file2)
    
    if not file1.exists():
        print(f"Error: {file1} does not exist")
        sys.exit(1)
    if not file2.exists():
        print(f"Error: {file2} does not exist")
        sys.exit(1)
    
    comparer = DiffBasedComparer(width=args.width)
    comparer.compare_files(file1, file2)

if __name__ == "__main__":
    main()
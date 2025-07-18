#!/usr/bin/env python3
"""
PPO Log Comparison Tool

Compares two PPO cycle debug logs side-by-side by:
1. Preprocessing to remove numbers
2. Using diff to find structural differences
3. Displaying aligned output with original values
"""

import re
import sys
import subprocess
import tempfile
import os
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import argparse

class PPOLogComparator:
    def __init__(self, width: int = 80):
        self.width = width
        self.number_pattern = re.compile(r'[-+]?\d*\.?\d+([eE][-+]?\d+)?')
        self.tensor_shape_pattern = re.compile(r'torch\.Size\(\[[^\]]*\]\)')
        self.list_pattern = re.compile(r'\[[-\d., ]+\]')
        
    def preprocess_line(self, line: str) -> str:
        """Remove numbers from a line for structural comparison"""
        # First handle special patterns
        line = self.tensor_shape_pattern.sub('torch.Size([<NUM>])', line)
        line = self.list_pattern.sub('[<NUM>]', line)
        # Then handle remaining numbers
        line = self.number_pattern.sub('<NUM>', line)
        return line
    
    def preprocess_file(self, filepath: Path) -> Path:
        """Create a preprocessed version of the file with numbers removed"""
        temp_file = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='_preprocessed.txt')
        
        with open(filepath, 'r') as f:
            for line in f:
                preprocessed = self.preprocess_line(line.rstrip())
                temp_file.write(preprocessed + '\n')
        
        temp_file.close()
        return Path(temp_file.name)
    
    def run_diff(self, file1: Path, file2: Path) -> List[str]:
        """Run diff on two files and return the output lines"""
        try:
            result = subprocess.run(
                ['diff', '--unified=0', str(file1), str(file2)],
                capture_output=True,
                text=True
            )
            return result.stdout.splitlines()
        except subprocess.CalledProcessError as e:
            print(f"Error running diff: {e}")
            return []
    
    def parse_diff_output(self, diff_lines: List[str]) -> Dict[str, List[Tuple[int, int]]]:
        """Parse diff output to get line mappings"""
        changes = {
            'matched': [],    # (line1, line2) pairs that match
            'only_file1': [], # line numbers only in file1
            'only_file2': []  # line numbers only in file2
        }
        
        # Simple parsing - this is a basic implementation
        # In practice, we'll use a different approach for better alignment
        return changes
    
    def align_files(self, file1_lines: List[str], file2_lines: List[str], 
                   preprocessed1: Path, preprocessed2: Path) -> List[Tuple[Optional[str], Optional[str]]]:
        """Align two files based on their preprocessed versions"""
        # Create lookup of preprocessed lines
        prep1_lines = []
        prep2_lines = []
        
        with open(preprocessed1, 'r') as f:
            prep1_lines = [line.rstrip() for line in f]
        with open(preprocessed2, 'r') as f:
            prep2_lines = [line.rstrip() for line in f]
        
        # Use dynamic programming for alignment (simplified version)
        aligned_pairs = []
        i, j = 0, 0
        
        while i < len(file1_lines) or j < len(file2_lines):
            if i >= len(file1_lines):
                # Remaining lines from file2
                aligned_pairs.append((None, file2_lines[j]))
                j += 1
            elif j >= len(file2_lines):
                # Remaining lines from file1
                aligned_pairs.append((file1_lines[i], None))
                i += 1
            elif i < len(prep1_lines) and j < len(prep2_lines) and prep1_lines[i] == prep2_lines[j]:
                # Lines match structurally
                aligned_pairs.append((file1_lines[i], file2_lines[j]))
                i += 1
                j += 1
            else:
                # Try to find next match within a window
                found_match = False
                window_size = 5
                
                # Check if file1[i] appears soon in file2
                for k in range(j, min(j + window_size, len(prep2_lines))):
                    if i < len(prep1_lines) and prep1_lines[i] == prep2_lines[k]:
                        # Add unmatched file2 lines
                        for m in range(j, k):
                            aligned_pairs.append((None, file2_lines[m]))
                        # Add the match
                        aligned_pairs.append((file1_lines[i], file2_lines[k]))
                        i += 1
                        j = k + 1
                        found_match = True
                        break
                
                if not found_match:
                    # Check if file2[j] appears soon in file1
                    for k in range(i, min(i + window_size, len(prep1_lines))):
                        if j < len(prep2_lines) and prep2_lines[j] == prep1_lines[k]:
                            # Add unmatched file1 lines
                            for m in range(i, k):
                                aligned_pairs.append((file1_lines[m], None))
                            # Add the match
                            aligned_pairs.append((file1_lines[k], file2_lines[j]))
                            i = k + 1
                            j += 1
                            found_match = True
                            break
                
                if not found_match:
                    # No match found, advance both
                    if i < len(file1_lines) and j < len(file2_lines):
                        aligned_pairs.append((file1_lines[i], file2_lines[j]))
                        i += 1
                        j += 1
                    elif i < len(file1_lines):
                        aligned_pairs.append((file1_lines[i], None))
                        i += 1
                    else:
                        aligned_pairs.append((None, file2_lines[j]))
                        j += 1
        
        return aligned_pairs
    
    def format_line(self, text: Optional[str], width: int) -> str:
        """Format a line to fit within the given width"""
        if text is None:
            return ' ' * width
        
        if len(text) > width:
            return text[:width-3] + '...'
        else:
            return text.ljust(width)
    
    def display_comparison(self, aligned_pairs: List[Tuple[Optional[str], Optional[str]]], 
                          file1_name: str, file2_name: str,
                          filter_pattern: Optional[str] = None):
        """Display the aligned comparison"""
        # Calculate column widths
        total_width = 160  # Total terminal width
        divider = ' | '
        col_width = (total_width - len(divider)) // 2
        
        # Header
        header1 = f"IGE Log ({Path(file1_name).name})"
        header2 = f"DNNE Log ({Path(file2_name).name})"
        print(f"{header1:<{col_width}}{divider}{header2:<{col_width}}")
        print('=' * col_width + divider + '=' * col_width)
        
        # Display aligned lines
        for line1, line2 in aligned_pairs:
            # Apply filter if specified
            if filter_pattern:
                if line1 and filter_pattern not in line1 and line2 and filter_pattern not in line2:
                    continue
            
            # Skip empty lines on both sides
            if (not line1 or not line1.strip()) and (not line2 or not line2.strip()):
                continue
            
            # Format and display
            formatted1 = self.format_line(line1, col_width)
            formatted2 = self.format_line(line2, col_width)
            
            # Add color if lines differ
            if line1 != line2:
                if line1 is None:
                    # Line only in file2 (green)
                    print(f"{formatted1}{divider}\033[32m{formatted2}\033[0m")
                elif line2 is None:
                    # Line only in file1 (red)
                    print(f"\033[31m{formatted1}\033[0m{divider}{formatted2}")
                else:
                    # Lines differ (yellow)
                    print(f"\033[33m{formatted1}\033[0m{divider}\033[33m{formatted2}\033[0m")
            else:
                # Lines match
                print(f"{formatted1}{divider}{formatted2}")
    
    def compare_files(self, file1: Path, file2: Path, filter_pattern: Optional[str] = None):
        """Main comparison function"""
        print(f"Comparing {file1} and {file2}...")
        
        # Read original files
        with open(file1, 'r') as f:
            file1_lines = [line.rstrip() for line in f]
        with open(file2, 'r') as f:
            file2_lines = [line.rstrip() for line in f]
        
        # Preprocess files
        print("Preprocessing files...")
        preprocessed1 = self.preprocess_file(file1)
        preprocessed2 = self.preprocess_file(file2)
        
        try:
            # Align files
            print("Aligning files...")
            aligned_pairs = self.align_files(file1_lines, file2_lines, preprocessed1, preprocessed2)
            
            # Display comparison
            print("\nComparison Results:")
            print("-" * 160)
            self.display_comparison(aligned_pairs, str(file1), str(file2), filter_pattern)
            
            # Summary statistics
            print("\n" + "-" * 160)
            matched = sum(1 for l1, l2 in aligned_pairs if l1 and l2 and self.preprocess_line(l1) == self.preprocess_line(l2))
            only_file1 = sum(1 for l1, l2 in aligned_pairs if l1 and not l2)
            only_file2 = sum(1 for l1, l2 in aligned_pairs if not l1 and l2)
            different = sum(1 for l1, l2 in aligned_pairs if l1 and l2 and self.preprocess_line(l1) != self.preprocess_line(l2))
            
            print(f"\nSummary:")
            print(f"  Matched lines: {matched}")
            print(f"  Lines only in IGE: {only_file1}")
            print(f"  Lines only in DNNE: {only_file2}")
            print(f"  Different lines: {different}")
            
        finally:
            # Cleanup
            os.unlink(preprocessed1)
            os.unlink(preprocessed2)


def main():
    parser = argparse.ArgumentParser(description='Compare two PPO debug log files side by side')
    parser.add_argument('file1', help='First log file (typically IGE)')
    parser.add_argument('file2', help='Second log file (typically DNNE)')
    parser.add_argument('--filter', '-f', help='Filter pattern (e.g., "DNNE_DEBUG")')
    parser.add_argument('--width', '-w', type=int, default=80, help='Column width')
    
    args = parser.parse_args()
    
    file1 = Path(args.file1)
    file2 = Path(args.file2)
    
    if not file1.exists():
        print(f"Error: {file1} does not exist")
        sys.exit(1)
    if not file2.exists():
        print(f"Error: {file2} does not exist")
        sys.exit(1)
    
    comparator = PPOLogComparator(width=args.width)
    comparator.compare_files(file1, file2, args.filter)


if __name__ == '__main__':
    main()
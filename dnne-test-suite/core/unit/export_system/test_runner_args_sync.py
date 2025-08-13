"""
Test that runner_args.json stays in sync with arg_parser.tpl
This test MUST pass for exports to work correctly.
"""

import unittest
import json
import os
import re
from pathlib import Path

class TestRunnerArgsSync(unittest.TestCase):
    """Ensure UI configuration matches command-line arguments"""
    
    @classmethod
    def setUpClass(cls):
        """Find the template files"""
        # Navigate from dnne-test-suite/core/unit/export_system/ to export_system/templates/framework
        cls.template_dir = Path(__file__).parent.parent.parent.parent.parent / 'export_system/templates/framework'
        cls.arg_parser_path = cls.template_dir / 'arg_parser.tpl'
        cls.runner_args_path = cls.template_dir / 'runner_args.json'
    
    def test_files_exist(self):
        """Both configuration files must exist"""
        self.assertTrue(self.arg_parser_path.exists(), 
                       f"arg_parser.tpl not found at {self.arg_parser_path}")
        self.assertTrue(self.runner_args_path.exists(),
                       f"runner_args.json not found at {self.runner_args_path}")
    
    def test_args_in_sync(self):
        """All parser arguments must have UI configuration"""
        parser_args = self._extract_parser_arguments()
        ui_args = self._load_ui_arguments()
        
        # Arguments intentionally omitted from UI:
        # - headless: default behavior, saves UI space
        # - dnne_profiling: too obscure for normal users
        # - debug: redundant with verbose (--debug == --verbose DEBUG)
        # - verbose: handled differently in UI as "logging" with select options
        intentionally_omitted = {'headless', 'dnne_profiling', 'debug', 'verbose'}
        
        # Check for missing arguments in UI (excluding intentional omissions)
        missing_in_ui = parser_args - ui_args - intentionally_omitted
        self.assertEqual(missing_in_ui, set(),
            f"Arguments in arg_parser.tpl but missing from runner_args.json: {missing_in_ui}\n"
            f"Please update runner_args.json with UI configuration for these arguments"
        )
        
        # Check for extra arguments in UI (possibly removed from parser)
        # Note: 'logging' in UI maps to 'verbose' in parser
        ui_only_mappings = {'logging'}  # These UI args map to different parser args
        extra_in_ui = ui_args - parser_args - ui_only_mappings
        self.assertEqual(extra_in_ui, set(),
            f"Arguments in runner_args.json but not in arg_parser.tpl: {extra_in_ui}\n"
            f"These arguments may have been removed - please clean up runner_args.json"
        )
    
    def test_timestamp_order(self):
        """runner_args.json should be newer than or same age as arg_parser.tpl"""
        parser_mtime = os.path.getmtime(self.arg_parser_path)
        ui_mtime = os.path.getmtime(self.runner_args_path)
        
        # Only warn, don't fail - the content test above is the real validator
        if parser_mtime > ui_mtime:
            import time
            parser_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(parser_mtime))
            ui_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(ui_mtime))
            print(f"\n⚠️  Warning: arg_parser.tpl ({parser_time}) is newer than "
                  f"runner_args.json ({ui_time})")
    
    def test_switch_format_consistency(self):
        """Verify switch names match expected patterns"""
        with open(self.runner_args_path, 'r') as f:
            ui_config = json.load(f)
        
        for arg_name, config in ui_config['arguments'].items():
            switch = config.get('switch', '')
            
            # Switch should start with -- or -
            self.assertTrue(switch.startswith('--') or switch.startswith('-'),
                          f"Invalid switch format for {arg_name}: {switch}")
            
            # Special case: logging in UI maps to --verbose in arg parser
            if arg_name == 'logging' and switch == '--verbose':
                continue  # This is correct
            
            # Switch should relate to argument name (with underscores as hyphens)
            expected_switch = '--' + arg_name.replace('_', '-')
            if not switch.startswith('-') or len(switch) > 2:  # Not a short flag
                # Allow some flexibility for switches that don't exactly match
                # (e.g., --dnne-profiling for dnne_profiling)
                self.assertTrue(
                    switch == expected_switch or 
                    switch.replace('--', '').replace('-', '_') == arg_name,
                    f"Switch {switch} doesn't match expected pattern for {arg_name}"
                )
    
    def _extract_parser_arguments(self):
        """Extract argument names from arg_parser.tpl"""
        with open(self.arg_parser_path, 'r') as f:
            content = f.read()
        
        # Find all add_argument calls using regex
        # Matches: parser.add_argument('--name', ...) or ('-n', '--name', ...)
        pattern = r"parser\.add_argument\((.*?)\)"
        matches = re.findall(pattern, content, re.DOTALL)
        
        arguments = set()
        for match in matches:
            # Extract the argument names (could be '--name' or '-n', '--name')
            arg_pattern = r'[\'"](-{1,2}[\w-]+)[\'"]'
            arg_matches = re.findall(arg_pattern, match)
            
            for arg in arg_matches:
                if arg.startswith('--'):
                    # Convert --kebab-case to snake_case for matching
                    name = arg[2:].replace('-', '_')
                    arguments.add(name)
        
        return arguments
    
    def _load_ui_arguments(self):
        """Load argument names from runner_args.json"""
        with open(self.runner_args_path, 'r') as f:
            config = json.load(f)
        
        return set(config['arguments'].keys())
    
    def test_all_arguments_have_required_fields(self):
        """Verify each argument has all required fields in runner_args.json"""
        with open(self.runner_args_path, 'r') as f:
            ui_config = json.load(f)
        
        required_fields = {'switch', 'type', 'label', 'description'}
        
        for arg_name, config in ui_config['arguments'].items():
            missing_fields = required_fields - set(config.keys())
            self.assertEqual(missing_fields, set(),
                f"Argument '{arg_name}' missing required fields: {missing_fields}")
    
    def test_argument_order_completeness(self):
        """Verify argument_order contains all arguments"""
        with open(self.runner_args_path, 'r') as f:
            ui_config = json.load(f)
        
        args_in_dict = set(ui_config['arguments'].keys())
        args_in_order = set(ui_config.get('argument_order', []))
        
        missing_from_order = args_in_dict - args_in_order
        self.assertEqual(missing_from_order, set(),
            f"Arguments missing from argument_order list: {missing_from_order}")
        
        extra_in_order = args_in_order - args_in_dict
        self.assertEqual(extra_in_order, set(),
            f"Extra arguments in argument_order list: {extra_in_order}")

if __name__ == '__main__':
    unittest.main()
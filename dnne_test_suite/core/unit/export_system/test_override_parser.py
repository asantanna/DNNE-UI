"""
Unit tests for the override_parser module

Tests the parsing of --override command line arguments for runtime
configuration of node parameters.
"""

import pytest
import sys
from pathlib import Path

# Add export_system templates to path for import
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent / "export_system" / "templates"))

from framework.override_parser import parse_override_args, parse_single_override, parse_value


class TestParseValue:
    """Test the parse_value function"""
    
    def test_parse_boolean_true(self):
        """Test parsing boolean True values"""
        assert parse_value("True") is True
        assert parse_value("true") is True
        assert parse_value("TRUE") is True
    
    def test_parse_boolean_false(self):
        """Test parsing boolean False values"""
        assert parse_value("False") is False
        assert parse_value("false") is False
        assert parse_value("FALSE") is False
    
    def test_parse_integer(self):
        """Test parsing integer values"""
        assert parse_value("42") == 42
        assert parse_value("0") == 0
        assert parse_value("-123") == -123
    
    def test_parse_float(self):
        """Test parsing float values"""
        assert parse_value("3.14") == 3.14
        assert parse_value("0.001") == 0.001
        assert parse_value("-2.5") == -2.5
    
    def test_parse_quoted_string(self):
        """Test parsing quoted strings"""
        assert parse_value('"hello world"') == "hello world"
        assert parse_value("'single quotes'") == "single quotes"
        assert parse_value('"At End"') == "At End"
        assert parse_value('""') == ""  # Empty quoted string
    
    def test_parse_unquoted_string(self):
        """Test parsing unquoted strings"""
        assert parse_value("hello") == "hello"
        assert parse_value("end") == "end"
        assert parse_value("some_param") == "some_param"


class TestParseSingleOverride:
    """Test the parse_single_override function"""
    
    def test_valid_override(self):
        """Test parsing valid override expressions"""
        assert parse_single_override("56:checkpoint_enabled=True") == ("56", "checkpoint_enabled", True)
        assert parse_single_override("42:learning_rate=0.001") == ("42", "learning_rate", 0.001)
        assert parse_single_override('56:checkpoint_trigger_type="At End"') == ("56", "checkpoint_trigger_type", "At End")
        assert parse_single_override("38:batch_size=128") == ("38", "batch_size", 128)
    
    def test_override_with_spaces(self):
        """Test parsing with spaces around elements"""
        assert parse_single_override(" 56 : checkpoint_enabled = True ") == ("56", "checkpoint_enabled", True)
    
    def test_invalid_override_format(self):
        """Test invalid override formats return error messages"""
        result = parse_single_override("invalid:format")
        assert isinstance(result, str)
        assert "Invalid override format" in result
        
        result = parse_single_override("56=True")  # Missing param name
        assert isinstance(result, str)
        assert "Invalid override format" in result
        
        # Non-numeric strings are now valid (could be subsystem names)
        result = parse_single_override("training:param=value")
        assert not isinstance(result, str)  # Should be valid
        target, param, value = result
        assert target == "training"
        assert param == "param"
        assert value == "value"


class TestParseOverrideArgs:
    """Test the main parse_override_args function"""
    
    def test_empty_string(self):
        """Test parsing empty string returns empty configs"""
        configs, errors = parse_override_args("")
        assert configs == {}
        assert errors == []
        
        configs, errors = parse_override_args(None)
        assert configs == {}
        assert errors == []
    
    def test_single_override(self):
        """Test parsing single override"""
        configs, errors = parse_override_args("56:checkpoint_enabled=True")
        assert configs == {"56": {"checkpoint_enabled": True}}
        assert errors == []
    
    def test_multiple_overrides_same_node(self):
        """Test multiple parameters for same node"""
        configs, errors = parse_override_args("56:checkpoint_enabled=True,56:checkpoint_trigger_type=end")
        assert configs == {
            "56": {
                "checkpoint_enabled": True,
                "checkpoint_trigger_type": "end"
            }
        }
        assert errors == []
    
    def test_multiple_overrides_different_nodes(self):
        """Test parameters for different nodes"""
        configs, errors = parse_override_args("42:learning_rate=0.001,56:checkpoint_enabled=True,38:batch_size=64")
        assert configs == {
            "42": {"learning_rate": 0.001},
            "56": {"checkpoint_enabled": True},
            "38": {"batch_size": 64}
        }
        assert errors == []
    
    def test_quoted_strings_with_commas(self):
        """Test that quoted strings containing commas are handled correctly"""
        configs, errors = parse_override_args('56:message="Hello, world",42:value=123')
        assert configs == {
            "56": {"message": "Hello, world"},
            "42": {"value": 123}
        }
        assert errors == []
    
    def test_mixed_valid_and_invalid(self):
        """Test mixture of valid and invalid overrides"""
        configs, errors = parse_override_args("56:checkpoint_enabled=True,invalid:format,42:learning_rate=0.001")
        assert configs == {
            "56": {"checkpoint_enabled": True},
            "42": {"learning_rate": 0.001}
        }
        assert len(errors) == 1
        assert "Invalid override format" in errors[0]
    
    def test_edge_cases(self):
        """Test edge cases"""
        # Empty value
        configs, errors = parse_override_args("56:param=")
        assert configs == {"56": {"param": ""}}
        assert errors == []
        
        # Multiple equals signs
        configs, errors = parse_override_args("56:param=value=with=equals")
        assert configs == {"56": {"param": "value=with=equals"}}
        assert errors == []
        
        # Trailing comma
        configs, errors = parse_override_args("56:param=value,")
        assert configs == {"56": {"param": "value"}}
        assert errors == []
    
    def test_real_world_examples(self):
        """Test real-world usage examples"""
        # Enable checkpointing at end of run
        configs, errors = parse_override_args("56:checkpoint_enabled=True,56:checkpoint_trigger_type=end")
        assert configs == {
            "56": {
                "checkpoint_enabled": True,
                "checkpoint_trigger_type": "end"
            }
        }
        assert errors == []
        
        # Override multiple training parameters
        configs, errors = parse_override_args("68:learning_rate=0.01,38:batch_size=256,55:max_epochs=50")
        assert configs == {
            "68": {"learning_rate": 0.01},
            "38": {"batch_size": 256},
            "55": {"max_epochs": 50}
        }
        assert errors == []


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
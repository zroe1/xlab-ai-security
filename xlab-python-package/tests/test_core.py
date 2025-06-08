"""
Tests for the xlab.core module.
"""

import pytest
from unittest.mock import patch
from xlab.core import hello_world
from xlab import __version__


class TestHelloWorld:
    """Test cases for the hello_world function."""
    
    def test_hello_world_returns_correct_message(self):
        """Test that hello_world returns the correct message format."""
        expected_message = f"Hello world! You are using version {__version__} of the package"
        result = hello_world()
        assert result == expected_message
    
    @patch('builtins.print')
    def test_hello_world_prints_message(self, mock_print):
        """Test that hello_world prints the message."""
        expected_message = f"Hello world! You are using version {__version__} of the package"
        hello_world()
        mock_print.assert_called_once_with(expected_message)
    
    def test_hello_world_includes_version(self):
        """Test that hello_world includes the package version."""
        result = hello_world()
        assert __version__ in result
        assert "Hello world!" in result
        assert "You are using version" in result
        assert "of the package" in result


class TestPackageStructure:
    """Test cases for package structure and imports."""
    
    def test_version_is_string(self):
        """Test that the package version is a string."""
        assert isinstance(__version__, str)
        assert len(__version__) > 0
    
    def test_can_import_hello_world(self):
        """Test that hello_world can be imported from the main package."""
        from xlab import hello_world
        assert callable(hello_world) 
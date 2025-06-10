"""
Tests for the xlab package level functionality.
"""

import pytest
import xlab


class TestPackageImports:
    """Test cases for package imports and structure."""
    
    def test_package_has_version(self):
        """Test that the package has a version attribute."""
        assert hasattr(xlab, '__version__')
        assert isinstance(xlab.__version__, str)
    
    def test_package_has_hello_world(self):
        """Test that hello_world is available at package level."""
        assert hasattr(xlab, 'hello_world')
        assert callable(xlab.hello_world)
    
    def test_package_all_attribute(self):
        """Test that __all__ is properly defined."""
        assert hasattr(xlab, '__all__')
        assert 'hello_world' in xlab.__all__
        assert '__version__' in xlab.__all__
    
    def test_hello_world_execution(self):
        """Test that hello_world can be executed from package level."""
        result = xlab.hello_world()
        assert isinstance(result, str)
        assert xlab.__version__ in result 
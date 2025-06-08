"""
Setup configuration for the xlab package.
"""

from setuptools import setup, find_packages
import os

# Read the version from the package
def get_version():
    version_file = os.path.join(os.path.dirname(__file__), 'xlab', '__init__.py')
    with open(version_file, 'r') as f:
        for line in f:
            if line.startswith('__version__'):
                return line.split('=')[1].strip().strip('"\'')
    return "0.1.0"

# Read the README file for long description
def get_long_description():
    readme_path = os.path.join(os.path.dirname(__file__), 'README.md')
    if os.path.exists(readme_path):
        with open(readme_path, 'r', encoding='utf-8') as f:
            return f.read()
    return ""

setup(
    name="xlab-security",
    version=get_version(),
    author="XLab Team",
    author_email="contact@xlab.ai",
    description="A Python package for AI security and helper functions",
    long_description=get_long_description(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/xlab-security",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.7",
    install_requires=[
        # Add your dependencies here
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=21.0",
            "flake8>=3.9",
        ],
    },
) 
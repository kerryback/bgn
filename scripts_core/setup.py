"""Setup script for Refactor2."""

from setuptools import setup, find_packages
from pathlib import Path

# Read README
readme_file = Path(__file__).parent / 'README.md'
long_description = readme_file.read_text() if readme_file.exists() else ''

setup(
    name='asset-pricing-refactor2',
    version='2.0.0',
    description='Refactored asset pricing simulation with unified panel generation and analysis',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='',
    author_email='',
    python_requires='>=3.9',
    packages=find_packages(),
    install_requires=[
        'numpy>=1.21',
        'pandas>=1.3',
        'scipy>=1.7',
        'scikit-learn>=1.0',
        'joblib>=1.0',
        'tqdm>=4.62',
    ],
    extras_require={
        'dev': [
            'pytest>=6.0',
            'pytest-cov>=2.0',
        ],
        'yaml': [
            'pyyaml>=5.0',
        ]
    },
    entry_points={
        'console_scripts': [
            'run-simulation=main:main',
        ],
    },
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
    ],
)

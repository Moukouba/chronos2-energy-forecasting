"""
Setup script for energy_forecast package
"""

from setuptools import setup, find_packages

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

with open("README.md") as f:
    long_description = f.read()

setup(
    name="energy_forecast",
    version="1.0.0",
    author="Energy Forecasting Team",
    author_email="energy@forecasting.com",
    description="Production-ready energy forecasting pipeline using Chronos2",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/energy/forecasting",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Information Analysis",
    ],
    python_requires=">=3.9",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "energy-forecast=main:main",
        ],
    },
    include_package_data=True,
    package_data={
        "energy_forecast": ["config/*.yaml", "config/*.json"],
    },
)

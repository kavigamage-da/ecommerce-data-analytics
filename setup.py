# setup.py
from setuptools import setup, find_packages

# Read requirements.txt
with open("requirements.txt") as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]

setup(
    name="ecommerce_data_analytics",
    version="0.1.0",
    packages=find_packages(),  # Automatically include all packages in src/
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "eda=src.pipeline:main",  # Optional CLI entry point for your pipeline
        ],
    },
    description="FAANG-level E-Commerce Data Analytics Project",
    author="Kavi Gamage",
    license="MIT",
    python_requires=">=3.10",
)
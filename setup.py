from setuptools import setup, find_packages

setup(
    name="txm_processor",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "scikit-image",
        "rich",
        "xrmreader",
        "matplotlib",  # Added for image saving
    ],
    entry_points={
        "console_scripts": [
            "txm-process=txm_processor.cli:main",
        ],
    },
    author="Nardos Estifanos",
    author_email="nardosnegussie519461@gmail.com",
    description="A package for processing TXM files with efficient view extraction and patchification",
    keywords="txm, 3D volume, image processing",
    python_requires=">=3.7",
)

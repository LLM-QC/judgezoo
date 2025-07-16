from setuptools import setup, find_packages

setup(
    name="judgezoo",
    version="0.1.1",
    description="A collection of judges for evaluating language model generations",
    author="Tim Beyer",
    author_email="tim.beyer@tum.de",
    packages=find_packages(),
    install_requires=[
        "torch>=2.3.1",
        "transformers>=4.45.0",
        "numpy",
        "openai",
    ],
    extras_require={
        "test": [
            "pytest",
            "pytest-cov",
            "pytest-mock",
        ],
    },
)

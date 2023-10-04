from setuptools import setup, find_packages

setup(
    name="sentiment-intention-analysis",
    version="0.1.0",
    packages=find_packages(include=["src", "src.*","src.analysis", "src.analysis.*", "src.utils", "src.utils.*"]),
    install_requires=[
        "transformers",
        "pandas",
    ],
)

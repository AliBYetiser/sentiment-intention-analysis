from setuptools import setup, find_packages

setup(
    name="sentiment-intention-analysis",
    version="0.1",
    packages=find_packages(include=["src", "src.*","src.analysis", "src.analysis.*", "src.utils", "src.utils.*"]),
    license="MIT License",
    author="AliBYetiser",
    author_email="abyetiser@gmail.com",
    description="Toolkit for sentiment analysis and intention identification in text conversations between agents and customers using zero-shot techniques.",
    install_requires=[
        "transformers",
        "pandas",
    ],
)

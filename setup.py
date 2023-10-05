from distutils.core import setup

setup(
    name='sentiment-intention-analysis',
    version='0.1.0',
    packages=['sentiment_intention_analysis', 'sentiment_intention_analysis.utils', 'sentiment_intention_analysis.analysis'],
    license='MIT',
    author='AliBYetiser',
    author_email='abyetiser@gmail.com',
    description='Toolkit for sentiment analysis and intention identification in text conversations between agents and customers using zero-shot techniques.',
    requires=[
        "transformers",
        "pandas",
        "torch",
        "Flask"
    ]
)

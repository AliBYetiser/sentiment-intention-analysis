# sentiment-intention-analysis
 Toolkit for sentiment analysis and intention identification in text conversations between agents and customers using zero-shot techniques.

Runs zero-shot classification with bart-large-mnli model on Google App Engine.

To run locally, clone this repository and then in '/resources' clone bart-large-mnli from huggingface with 
```
git lfs install
git clone https://huggingface.co/facebook/bart-large-mnli
```
then run app.py, and in a browser access 

http://127.0.0.1:8080/sample_transcript or  [http://127.0.0.1:8080/predict?input=*transcript*](http://127.0.0.1:8080/predict?input=transcript) 

where *transcript* is your text input for sentiment and intention analysis in the context of an agent-customer interaction.

Equivalently, install locally as a package to run app main, or access functions:
```
$ pip install .
$ python
>> from sentiment_intention_analysis import app
```

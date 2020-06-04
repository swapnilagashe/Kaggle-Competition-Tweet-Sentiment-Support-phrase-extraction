# Kaggle-Tweet-Sentiment-Support-phrase-extraction

In this project the aim is to predict the context of a sentiment for a tweet. Consider the following example:
Tweet : "I didn't like this song, has anyone heard other songs from eminem?"
Sentiment : Negative
Context(which is to be predicted) : "I didn't like this song"

Dataset description:

Training dataset --> contains the textID, tweet ('text'), sentiment(Positive, Negative or Neutral), context(selected_text)
Test dataset --> contains the textID, tweet('text'), sentiment(Positive, Negative or Neutral)
Accuracy Metric used --> Jaccard Score

Note: I have not done any sort of preprocessing to the text because our predictions are also raw text (Preprocessing the text will badly affect our accuracy)

I have tried multiple approaches to get the desired predictions. Specifically:
1. Spacy Custom NER model (Named Entity Recognition approach)
2. BERT (Question/Answering approach)
3. RoBERTa (Question/Answering approach)

1. Spacy NER model : Cool thing about spacy's Named Entity Recognition is that we can train the model on our own dataset according to our needs

Here we treat the tweet(text) as input text and context(selected_text) as the Named Entity.

One thing to note here is that the prediction accuracy significantly improves when we train 3 seperate models for each type of sentiment. 

This model achieved a leader board score of around 0.65 which is not bad considering the top score is around 0.73. But If we wan't to improve the score further, we will need a different and better approach. Hence we start looking at other models like BERT, RoBERTa etc.,


2. RoBERTa : 
I have used the RoBERTa base model as the large model is very large both in terms of the size and training time(and I think it will only cause incremental increase in the accuracy).

Here we used the pretrained RoBERTa model and modify the head (few top layers of the Neural Network architecture) according to our need (Question/Answering).

Note: we have to provide a reference text whenever we want to get answer to some question from a model.
Therefore, We will treat the textID as Question ID, Tweet as a reference, Sentiment as a Question and our Context(to be predicted) as the answer.


I will post the final accuracy score of the model soon, but I think it will be somewhere around 0.7




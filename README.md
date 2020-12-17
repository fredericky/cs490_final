# Objective
Tweet sentiment analysis, the training/test data are from the Kaggle (https://www.kaggle.com/c/tweet-sentiment-extraction/data).
* Use Pytorch Dataset/Dataloader to pre-process the text data
* Understand the Transformers and BERT
* Use Transfer learning to build sentiment classisifer with Hugging Face Transformers libaray
* Evaluate the model on the test data

# Data Analysis
![](images/data_samples.png)
The training data set has 27481 rows, and each row includes a textID, text, selected_text and sentiment. For simplifying this classification task, its input is text and the label is sentiment (neutral, negative and positive). Here is the sentiment distribution of the training data.
![](images/sentiment_distribution.png)

# Data Pre-processing

# Model & Training

# Training

# Evalution

# References
* [Sentiment Analysis with BERT and Transformers by Hugging Face using PyTorch and Python](https://curiousily.com/posts/sentiment-analysis-with-bert-and-hugging-face-using-pytorch-and-python/)
* [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
* [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805)

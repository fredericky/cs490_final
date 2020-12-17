# Objective
Tweet sentiment analysis, the training/test data are from the [Kaggle](https://www.kaggle.com/c/tweet-sentiment-extraction/data).
* Use Pytorch Dataset/Dataloader to pre-process the text data
* Understand the Transformers and BERT
* Use Transfer learning to build sentiment classisifer with Hugging Face Transformers library
* Evaluate the model on the test data

# Data Analysis
![](images/data_samples.png)
The training data set has 27481 rows, and each row includes a textID, text, selected_text and sentiment. For simplifying this classification task, its input is text and the label is sentiment (neutral, negative and positive). Here is the sentiment distribution of the training data.
![](images/sentiment_distribution.png)

# Data Pre-processing
We will do the following per BERT requirements via Hugging Face Transformers library in this section.
* Add special tokens to separate the sentences
* Choose a fixed sequence length and add paddings for those length less than the choose length
* Create the attention mask

Specifically, [BertTokenizer](https://huggingface.co/transformers/model_doc/bert.html#berttokenizer) is for helping with word embeddings. It can be initialized with pre-trained model.
```
tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
```
Also this class provides the special tokens like
* `[SEP]`: the separator token, which is used when building a sequence from multiple sequences
* `[CLS]`: the classifier token which is used when doing sequence classification
* `[PAD]`: the token used for padding, for example when batching sequences of different lengths

BERT works with fixed length sequences, so we need to choose a length, here is the text length distribution of the training data.
![](images/length_distribution.png)
From the above, `150` is a good number.

Here is the data set abstraction by extending the Pytorch `Dataset` class.
```
MAX_LENGTH = 150
SENTIMENT = {
    'positive' : 0,
    'neutral': 1,
    'negative': 2
}

class TweetsDataset(Dataset):
  def __init__(self, tweets, labels):
    self.tweets = tweets
    self.labels = labels
  
  def __len__(self):
    return len(self.tweets)
  
  def __getitem__(self, i):
    tweet = self.tweets[i]
    label = self.labels[i]
    encoding = tokenizer.encode_plus(
        str(tweet), 
        add_special_tokens=True, 
        max_length=MAX_LENGTH, 
        return_token_type_ids=False, 
        pad_to_max_length=True,
        return_attention_mask=True,
        return_tensors='pt'
    )
    return {
        'tweet': tweet,
        'label': torch.tensor(SENTIMENT[label], dtype=torch.long),
        'input_ids': encoding['input_ids'].flatten(),
        'attention_mask': encoding['attention_mask'].flatten(),
    }
```
Also we use 20% of the training data for the model validation.
```
df = pd.read_csv("train.csv")
df_test = pd.read_csv('test.csv')
df_train, df_val = train_test_split(df, test_size=0.2, random_state=42)
```
Note: `train.csv` and `test.csv` is from the [Kaggle](https://www.kaggle.com/c/tweet-sentiment-extraction/data).

# Model & Training

# Training

# Evalution

# References
* [Sentiment Analysis with BERT and Transformers by Hugging Face using PyTorch and Python](https://curiousily.com/posts/sentiment-analysis-with-bert-and-hugging-face-using-pytorch-and-python/)
* [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
* [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805)

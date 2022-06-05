import os
import nltk
import argparse
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import TweetTokenizer
from kaggle.api.kaggle_api_extended import KaggleApi

hol_data_link = 'https://github.com/t-davidson/hate-speech-and-offensive-language/blob/master/data/labeled_data.csv?raw=true'
ethos_data_link = 'https://github.com/intelligence-csd-auth-gr/Ethos-Hate-Speech-Dataset/blob/master/ethos/ethos_data/Ethos_Dataset_Binary.csv?raw=true'   

def preprocess_tweets(tweet):
    new_tweet = []
    tokenizer = TweetTokenizer()
    tweet = tokenizer.tokenize(tweet)
    for word in tweet:
        word = '@user' if word.startswith('@') and len(word) > 1 else word
        word = 'http' if word.startswith('http') else word
        new_tweet.append(word)
    return " ".join(new_tweet)


def remove_stopwords(text, stopwords):
  text = [word for word in text.split(' ') if word.lower() not in stopwords]
  return " ".join(text)


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--kaggle_username", type=str, default=None
    )
    parser.add_argument(
        "--kaggle_api_key", type=str, default=None
    )

    args = parser.parse_args()

    os.environ['KAGGLE_USERNAME'] = args.kaggle_username
    os.environ['KAGGLE_KEY'] = args.kaggle_api_key

    # Kaggle authorization
    api = KaggleApi()
    api.authenticate()

    # Download datasets
    api.dataset_download_file('rajkumarl/ruddit-jigsaw-dataset', file_name='Dataset/ruddit_with_text.csv', path="./data")
    api.competition_download_file('jigsaw-toxic-comment-classification-challenge', file_name='train.csv.zip', path="./data")

    # Read data
    twitter = pd.read_csv(hol_data_link, index_col = 0)
    ethos = pd.read_csv(ethos_data_link, sep = ';')
    ruddit = pd.read_csv('./data/ruddit_with_text.csv.zip')[['txt', 'offensiveness_score']]
    toxkaggle = pd.read_csv('./data/train.csv.zip')

    # Preprocess data
    nltk.download('stopwords')

    twitter.loc[twitter['class'] == 2, 'class'] = 1
    twitter = twitter.rename(columns={"tweet": "text", "class": "label"})
    twitter['text'] = twitter['text'].apply(preprocess_tweets) # remove Twitter specific parts 
    twitter = twitter[['text', 'label']]

    ethos = ethos.rename(columns={"comment": "text", "isHate": "label"})
    ethos = ethos.astype({'label': 'int32'})[['label', 'text']]

    ruddit = ruddit.loc[ruddit.txt != '[deleted]']
    ruddit = ruddit.loc[ruddit.txt != '[removed]']
    ruddit = ruddit.rename(columns={'txt':'text', 'offensiveness_score':'label'})
    ruddit['label'] = round((ruddit["label"] + 1.) / 2., 0)

    toxkaggle['label'] = round(toxkaggle.loc[:,'toxic':].sum(axis=1)/6, 0)
    toxkaggle = toxkaggle.rename(columns={'comment_text':'text'})
    toxkaggle = toxkaggle[['text', 'label']]

    train_data = pd.concat([ruddit, twitter, ethos, toxkaggle]) 

    # Remove stopwords
    en_stopwords = stopwords.words('english')
    train_data['text'] = train_data['text'].apply(remove_stopwords, args=(en_stopwords,))
    
    return train_data


if __name__ == "__main__":
    main()
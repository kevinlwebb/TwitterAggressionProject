import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(filepath):
    if 'json' in filepath:
        df = pd.read_json(filepath, lines= True)
        file_type = "json"
    else:
        df = pd.read_csv(filepath)
        file_type = "csv"
    
    return df, file_type


def clean_data(df, file_type):
    if file_type == "csv":
        df['cleaned_tweet'] = df.tweet.apply(lambda x: ' '.join([word for word in x.split() if not word.startswith('@')]))
        df['cleaned_tweet'] = df.cleaned_tweet.str.replace("#","")
    else:
        df["label"] = df.annotation.apply(lambda x: x.get('label'))
        df["label"] = df.label.apply(lambda x: x[0])
        df["cleaned_tweet"] = df["content"]
        df = df[["cleaned_tweet","label"]]

    return df


def save_data(df, database_filename):
    engine = create_engine('sqlite:///' + database_filename)
    df.to_sql("tweets", engine, index=False)


def main():
    if len(sys.argv) == 3:

        tweets_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    TWEETS: {}'.format(tweets_filepath))
        df, file_type = load_data(tweets_filepath)

        print('Cleaning data...')
        df = clean_data(df, file_type)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the tweets'\
              'dataset as the first argument, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the second argument. \n\nExample: python process_data.py '\
              'tweet_train.csv TweetSentiment.db')


if __name__ == '__main__':
    main()
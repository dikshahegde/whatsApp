from urlextract import URLExtract
from wordcloud import WordCloud
from collections import Counter
import emoji 
import nltk
import spacy
import re
from nltk.sentiment import  SentimentIntensityAnalyzer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation


nlp = spacy.load('en_core_web_sm')
#nltk.download('maxent_ne_chunker')
#nltk.download('vader_lexicon')
#nltk.download('words')
#nltk.download('stopwords')
#nltk.download('punkt')
ext=URLExtract()
import pandas as pd
def fetch_stats(selected_user,df):
    if selected_user != 'Overall':
        df=df[df['user']==selected_user]
    num_messages =df.shape[0]
    words=[]
    for message in df['message']:
        words.extend(message.split())

    num_media=df[df['message']=='<Media omitted>'].shape[0] 
    links=[]
    for message in df['message']:
        links.extend(ext.find_urls(message)) 
        
    return num_messages,len(words),num_media,len(links)

def most_busy_users(df):
    x=df['user'].value_counts().head()
    df= (df['user'].value_counts(normalize=True) * 100).round(2).reset_index()
    df.columns = ['name', 'percent']
    return x,df

def create_wordcloud(selected_user,df):

    f = open('stop_hinglish.txt', 'r')
    stop_words = f.read()

    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    temp = df[df['user'] != 'group_notification']
    temp = temp[temp['message'] != '<Media omitted>']
    temp = temp[temp['message'] != 'You deleted this message']

    def remove_stop_words(message):
        y = []
        for word in message.lower().split():
            if word not in stop_words:
                y.append(word)
        return " ".join(y)

    wc = WordCloud(width=500,height=500,min_font_size=10,background_color='white')
    temp['message'] = temp['message'].apply(remove_stop_words)
    df_wc = wc.generate(temp['message'].str.cat(sep=" "))
    return df_wc

def most_common_words(selected_user,df):

    f = open('stop_hinglish.txt','r')
    stop_words = f.read()

    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    temp = df[df['user'] != 'group_notification']
    temp = temp[temp['message'] != '<Media omitted>']

    words = []

    for message in temp['message']:
        for word in message.lower().split():
            if word not in stop_words:
                words.append(word)

    most_common_df = pd.DataFrame(Counter(words).most_common(20))
    return most_common_df


def emoji_helper(selected_user,df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    emojis = []
    for message in df['message']:
        emojis.extend([c for c in message if c in emoji.EMOJI_DATA])

    emoji_counter = Counter(emojis)
    emoji_df = pd.DataFrame(emoji_counter.most_common(), columns=['Emoji', 'Count'])

    return emoji_df

def daily_timeline(selected_user,df):

    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    daily_timeline = df.groupby('only_date').count()['message'].reset_index()

    return daily_timeline

def week_activity_map(selected_user,df):

    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    return df['day_name'].value_counts()

def monthly_timeline(selected_user,df):

    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    timeline = df.groupby(['year', 'month_num', 'Month']).count()['message'].reset_index()

    time = []
    for i in range(timeline.shape[0]):
        time.append(timeline['Month'][i] + "-" + str(timeline['year'][i]))

    timeline['time'] = time

    return timeline

def month_activity_map(selected_user,df):

    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    return df['Month'].value_counts()


def activity_heatmap(selected_user,df):

    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    user_heatmap = df.pivot_table(index='day_name', columns='period', values='message', aggfunc='count').fillna(0)

    return user_heatmap

import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

def sentiment_analysis(selected_user, df):
    sia = SentimentIntensityAnalyzer()
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]
    sentiments = []
    # Get sentiment scores for each message
    for message in df['message']:
        sentiments.append(sia.polarity_scores(message))

    # Convert the list of sentiment scores to a DataFrame
    sentiment_df = pd.DataFrame(sentiments)

    # Determine which sentiment is the highest for each message
    sentiment_df['max_sentiment'] = sentiment_df[['neg', 'neu', 'pos']].idxmax(axis=1)

    # Create a new DataFrame with the highest sentiment for each message
    sentiment_df_max = pd.DataFrame()
    sentiment_df_max['message'] = df['message']
    sentiment_df_max['sentiment'] = sentiment_df['max_sentiment']
    
    # Map sentiment to human-readable labels
    sentiment_df_max['sentiment'] = sentiment_df_max['sentiment'].map({
        'neg': 'Negative',
        'neu': 'Neutral',
        'pos': 'Positive'
    })

    # Calculate the mean sentiment scores
    mean_sentiment = sentiment_df[['neg', 'neu', 'pos']].mean().to_dict()

    # Determine overall sentiment based on the highest mean value
    overall_sentiment = max(mean_sentiment, key=mean_sentiment.get)
    if overall_sentiment == 'neg':
        overall_sentiment_label = 'Negative'
    elif overall_sentiment == 'neu':
        overall_sentiment_label = 'Neutral'
    elif overall_sentiment == 'pos':
        overall_sentiment_label = 'Positive'
    else:
        overall_sentiment_label = 'Unknown'

    return sentiment_df_max, overall_sentiment_label

def most_common_words(selected_user, df, top_n=5):
    f = open('stop_hinglish.txt', 'r')
    stop_words = f.read().splitlines()

    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    temp = df[df['user'] != 'group_notification']
    temp = temp[temp['message'] != '<Media omitted>']

    words = []
    for message in temp['message']:
        # Remove punctuation and split into words
        message = re.sub(r'[^\w\s]', '', message)
        for word in message.lower().split():
            if word not in stop_words:
                words.append(word)

    # Count the most common words
    word_counter = Counter(words)
    most_common_words = [word for word, count in word_counter.most_common(top_n)]

    return most_common_words
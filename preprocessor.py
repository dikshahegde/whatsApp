
import re
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

def preprocess(data):
    data = data.replace('\u202f', ' ')
    pattern = r'(\d{1,2}/\d{1,2}/\d{2,4},\s\d{1,2}:\d{2}\s[ap]m)'
    splits = re.split(pattern, data)[1:]
    
    dates = splits[::2]  # Dates are in even positions
    messages = splits[1::2]  # Messages are in odd positions
    
    print(f"Dates length: {len(dates)}, Messages length: {len(messages)}")  # Debug print

    cleaned_messages = [message.strip() for message in messages]
    
    if len(dates) == len(cleaned_messages):
        df = pd.DataFrame({'date': dates, 'user_message': cleaned_messages})
    else:
        print("Error: Mismatch between the number of dates and messages.")
        return None

    df['date'] = pd.to_datetime(df['date'], format='%d/%m/%y, %I:%M %p')

    def extract_name_or_phone(user_message):
        match = re.match(r'-\s(.+?):\s', user_message)
        if match:
            return match.group(1).strip()
        return "Unknown"

    df['user'] = df['user_message'].apply(extract_name_or_phone)

    def clean_message(user_message):
        return re.sub(r'-\s.+?:\s', '', user_message).strip()

    df['message'] = df['user_message'].apply(clean_message)

    df.drop(columns=['user_message'], inplace=True)

    # Remove rows with user 'Unknown'
    df = df[df['user'] != 'Unknown']
    df['only_date']=df['date'].dt.date
    df['year'] = df['date'].dt.year
    df['month_num']=df['date'].dt.month
    df['Month'] = df['date'].dt.month_name()
    df['day'] = df['date'].dt.day
    df['day_name'] = df['date'].dt.day_name()
    df['Hour'] = df['date'].dt.hour
    df['Minute'] = df['date'].dt.minute

    period = []
    for hour in df[['day_name','Hour']]['Hour']:
        if hour == 23:
            period.append(str( hour ) + "-" + str('00'))
        elif hour == 0:
            period.append(str('00') + "-" + str(hour + 1))
        else:
            period.append(str(hour) + "-" + str(hour + 1))

    df['period'] = period
    

    return df
    
    
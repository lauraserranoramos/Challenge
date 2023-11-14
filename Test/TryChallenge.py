import pandas as pd
import json
import requests
from bs4 import BeautifulSoup
from googletrans import Translator
import textblob
from textblob import TextBlob
from nltk.sentiment import SentimentIntensityAnalyzer
from langdetect import detect
#from tqdm.notebook import tqdm

json_file_path = r"C:\Users\User\PycharmProjects\pythonProject\Test\Data\data.json"

with open(json_file_path, 'r', encoding='utf-8') as file:
    data = json.load(file)

df = pd.read_json(json_file_path).rename_axis("articles")


def fetch_web_page_content(url):
    try:
        response = requests.get(url)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            text_content = soup.get_text()

            # TextBlob is designed to work with multiple languages, but it is more accurate in English. So we will translate the text in other languages:
            translator = Translator()
            translated_text = translator.translate(text_content)

            language = detect(translated_text.text)
            if language == 'en':
                return translated_text.text
            else:
                return "The text is not being translated to English"
            # return text_content
        else:
            return f"Failed to retrieve the web page. Status code: {response.status_code}"
    except Exception as e:
        return f"An error occurred: {str(e)}"


df['web_page_content'] = df['url'].apply(fetch_web_page_content).str.replace('\t', '').str.replace('\n', '').astype(
    'string')

df = df.drop(df[df['web_page_content'] == 'The text is not being translated to English'].index)
df = df[~df.apply(lambda row: row.astype(str).str.contains('An error occurred').any(), axis=1)]
df = df[~df.apply(lambda row: row.astype(str).str.contains('Failed to retrieve the web page').any(), axis=1)]

print(df)

#Total number of articles fetched:
print('Total number of articles fetched:', len(df))

#Name and number of unique countries:
print('Name of unique countries:', df['sourcecountry'].unique())
print('Number of unique countries:', len(df['sourcecountry'].unique()))

#Average length of the text:
def average_length(column):
    lengths = column.apply(len)
    return lengths.mean()

avg_length = average_length(df['web_page_content'])
print('Average length of the text:', avg_length)

#--------------------------------------------TEXTBLOB-----------------------------------------------------------------
print('TEXTBLOB')
#Obtaining polarity and subjectivity
def sentiment_polarity(df_column):
    polarity = TextBlob(df_column)
    return polarity.sentiment.polarity

df['sentiment_polarity'] = df['web_page_content'].apply(sentiment_polarity)

def sentiment_subjectivity(df_column):
    subjectivity = TextBlob(df_column)
    return subjectivity.sentiment.subjectivity

df['sentiment_subjectivity'] = df['web_page_content'].apply(sentiment_subjectivity)

#categorizying polarity from previous results
def categorize_sentiment(score):
    if score > 0.05:
        return 'Positive'
    elif score < -0.05:
        return 'Negative'
    else:
        return 'Neutral'
df['sentiment_category_polarity'] = df['sentiment_polarity'].apply(categorize_sentiment)

#categorizying subjectivity from previous results
def categorize_subjectivity(score):
    if score > 0.5:
        return 'yes'
    else:
        return 'No'
df['sentiment_Category_subjectivity'] = df['sentiment_subjectivity'].apply(categorize_subjectivity)
print(df)

#Obtaining the most positive
print(df.loc[df['sentiment_polarity'].idxmax()])

#Obtaining the most negative
print(df.loc[df['sentiment_polarity'].idxmin()])

#Obtaining percentage for each
sentiment_percentage = df['sentiment_category_polarity'].value_counts(normalize=True) * 100
print(sentiment_percentage)

#--------------------------------------------VADER-----------------------------------------------------------------
print('VADER')
sia = SentimentIntensityAnalyzer()
df.reset_index(inplace=True)

res = {}
for i, row in df.iterrows():
    text = row['web_page_content']
    article = row['articles']
    res[article] = sia.polarity_scores(text)

vaders = pd.DataFrame(res).T
vaders = vaders.reset_index().rename(columns={'index': 'articles'})
vaders = vaders.merge(df[['title','articles','web_page_content']], how='left', on = 'articles')

#categorize polarity
def categorize_polarity(row):
    if row['neu'] > 0.8:
        return 'neutral'
    elif row['pos'] > row['neg'] and row['neu'] < 0.99:
        return 'positive'
    else:
        return 'negative'

vaders['polarity'] = vaders.apply(categorize_polarity, axis=1)

#Most possitive
print('Most positive', vaders.loc[vaders['pos'].idxmax()])

#Most negative
print('Most negative', vaders.loc[vaders['neg'].idxmax()])

#Percentage
sentiment_percentage_vaders = vaders['polarity'].value_counts(normalize=True) * 100
print('Percentage', sentiment_percentage_vaders)

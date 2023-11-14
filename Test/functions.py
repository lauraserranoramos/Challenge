import pandas as pd
import requests
from bs4 import BeautifulSoup
from langdetect import detect
from textblob import TextBlob
from textblob.exceptions import TranslatorError
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
from googletrans import Translator
from langdetect import detect
from gdeltdoc import GdeltDoc, Filters
import pandas as pd
from datetime import datetime, timedelta






end_date = datetime.now()
start_date = end_date - timedelta(days=5)

f = Filters(
    keyword = "Artificial Intelligence",
    start_date = start_date.strftime('%Y%m%d'),
    end_date = end_date.strftime('%Y%m%d')
)

gd = GdeltDoc()

# Search for articles matching the filters
articles = gd.article_search(f)

# Get a timeline of the number of articles matching the filters

json_file_path = r"C:\Users\User\PycharmProjects\pythonProject\Test\Data\data5AI.json"

timeline = gd.timeline_search("timelinevol", f)

json_data = articles.to_json(orient='records')
with open(json_file_path, 'w') as json_file:
    json_file.write(json_data)


df = pd.read_json(json_file_path).rename_axis("articles")
def fetch_web_page_content(url):
    try:
        response = requests.get(url)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            text_content = soup.get_text()

            #  TextBlob is designed to work with multiple languages, but it is more accurate in English. So we will
            #  translate the text in other languages:
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

def clean_dataframe(df):
    df = df.drop(df[df['web_page_content'] == 'The text is not being translated to English'].index)
    df = df[~df.apply(lambda row: row.astype(str).str.contains('An error occurred').any(), axis=1)]
    df = df[~df.apply(lambda row: row.astype(str).str.contains('Failed to retrieve the web page').any(), axis=1)]
    return df

def get_basic_stats(df):
    print('Total number of articles fetched:', len(df))
    print('Name of unique countries:', df['sourcecountry'].unique())
    print('Number of unique countries:', len(df['sourcecountry'].unique()))
    avg_length = average_length(df['web_page_content'])
    print('Average length of the text:', avg_length)

def analyze_textblob_sentiment(df):
    df['sentiment_polarity'] = df['web_page_content'].apply(sentiment_polarity)
    df['sentiment_subjectivity'] = df['web_page_content'].apply(sentiment_subjectivity)
    df['sentiment_category_polarity'] = df['sentiment_polarity'].apply(categorize_sentiment)
    df['sentiment_Category_subjectivity'] = df['sentiment_subjectivity'].apply(categorize_subjectivity)
    print(df)

    print('Most positive', df.loc[df['sentiment_polarity'].idxmax()])
    print('Most negative', df.loc[df['sentiment_polarity'].idxmin()])

    sentiment_percentage = df['sentiment_category_polarity'].value_counts(normalize=True) * 100
    print(sentiment_percentage)

def analyze_vader_sentiment(df):
    sia = SentimentIntensityAnalyzer()
    df.reset_index(inplace=True)

    res = {}
    for i, row in df.iterrows():
        text = row['web_page_content']
        article = row['articles']
        res[article] = sia.polarity_scores(text)

    vaders = pd.DataFrame(res).T
    vaders = vaders.reset_index().rename(columns={'index': 'articles'})
    vaders = vaders.merge(df[['title','articles','web_page_content']], how='left', on='articles')

    vaders['polarity'] = vaders.apply(categorize_polarity, axis=1)

    print('Most positive', vaders.loc[vaders['pos'].idxmax()])
    print('Most negative', vaders.loc[vaders['neg'].idxmax()])

    sentiment_percentage_vaders = vaders['polarity'].value_counts(normalize=True) * 100
    print('Percentage', sentiment_percentage_vaders)

def average_length(column):
    lengths = column.apply(len)
    return lengths.mean()

def sentiment_polarity(text):
    try:
        polarity = TextBlob(text)
        return polarity.sentiment.polarity
    except Exception as e:
        print(f"Error in sentiment_polarity: {str(e)}")
        return 0.0

def sentiment_subjectivity(text):
    try:
        subjectivity = TextBlob(text)
        return subjectivity.sentiment.subjectivity
    except Exception as e:
        print(f"Error in sentiment_subjectivity: {str(e)}")
        return 0.0

def categorize_sentiment(score):
    if score > 0.05:
        return 'Positive'
    elif score < -0.05:
        return 'Negative'
    else:
        return 'Neutral'

def categorize_subjectivity(score):
    if score > 0.5:
        return 'Yes'
    else:
        return 'No'

def categorize_polarity(row):
    if row['neu'] > 0.8:
        return 'Neutral'
    elif row['pos'] > row['neg'] and row['neu'] < 0.99:
        return 'Positive'
    else:
        return 'Negative'

df['web_page_content'] = df['url'].apply(fetch_web_page_content).str.replace('\t', '').str.replace('\n', '').astype('string')
df = clean_dataframe(df)

# Display basic statistics
get_basic_stats(df)

# Analyze sentiment using TextBlob
analyze_textblob_sentiment(df)

# Analyze sentiment using VADER
analyze_vader_sentiment(df)
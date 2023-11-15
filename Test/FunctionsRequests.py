import pandas as pd
import requests
from bs4 import BeautifulSoup
from langdetect import detect
from textblob import TextBlob
from textblob.exceptions import TranslatorError
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from googletrans import Translator
from langdetect import detect
from gdeltdoc import GdeltDoc, Filters
from datetime import datetime, timedelta

def fetch_gdelt_articles(start_date, end_date, keyword="Artificial Intelligence"):
    f = Filters(
        keyword=keyword,
        start_date=start_date.strftime('%Y%m%d'),
        end_date=end_date.strftime('%Y%m%d')
    )

    gd = GdeltDoc()
    articles = gd.article_search(f)
    return articles

def save_articles_to_json(articles, json_file_path):
    json_data = articles.to_json(orient='records')
    with open(json_file_path, 'w') as json_file:
        json_file.write(json_data)

def read_articles_from_json(json_file_path):
    return pd.read_json(json_file_path).rename_axis("articles")

def fetch_web_page_content(url):
    try:
        response = requests.get(url)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            text_content = soup.get_text()

            translator = Translator()
            translated_text = translator.translate(text_content)

            language = detect(translated_text.text)
            if language == 'en':
                return translated_text.text
            else:
                return "The text is not being translated to English"
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
    # (unchanged)
    pass

def analyze_vader_sentiment(df):
    # (unchanged)
    pass

def average_length(column):
    lengths = column.apply(len)
    return lengths.mean()

# Other functions (sentiment_polarity, sentiment_subjectivity, categorize_sentiment, categorize_subjectivity, categorize_polarity) remain unchanged

def main():
    end_date = datetime.now()
    start_date = end_date - timedelta(days=5)
    json_file_path = r"C:\Users\User\PycharmProjects\pythonProject\Test\Data\data5AI.json"

    articles = fetch_gdelt_articles(start_date, end_date)
    save_articles_to_json(articles, json_file_path)

    df = read_articles_from_json(json_file_path)
    df['web_page_content'] = df['url'].apply(fetch_web_page_content).str.replace('\t', '').str.replace('\n', '').astype('string')
    df = clean_dataframe(df)

    # Display basic statistics
    get_basic_stats(df)

    # Analyze sentiment using TextBlob
    analyze_textblob_sentiment(df)

    # Analyze sentiment using VADER
    analyze_vader_sentiment(df)

if __name__ == "__main__":
    main()
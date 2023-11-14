import pandas as pd
import json
import requests
from bs4 import BeautifulSoup
from langdetect import detect, LangDetectException
from googletrans import Translator
from http.client import RemoteDisconnected
from ssl import SSLError
from requests.exceptions import RequestException, SSLError

json_file_path = 'data.json'

with open(json_file_path, 'r', encoding='utf-8') as file:
    data = json.load(file)

df = pd.read_json('data.json').rename_axis("articles")


def fetch_web_page_content(url):
    try:
        response = requests.get(url, verify=False)
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
    except (RequestException, SSLError, EOFError, LangDetectException) as e:
        return str(e)


df['web_page_content'] = df['url'].apply(fetch_web_page_content).str.replace('\t', '').str.replace('\n', '').astype(
    'string')
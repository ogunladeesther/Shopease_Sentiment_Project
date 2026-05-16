from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import spacy
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from src.data_ingestion import data_ingestion
import re
import logging
from Config.constant import Clean_data

# define logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s')


sentiment_data = data_ingestion()

class DataCleaning:
    def __init__(self):
        self._ensure_nltk()
        self.nlp = self._load_nlp()

    def _load_nlp(self) -> spacy.language.Language:
        for model in ("en_core_web_sm", "xx_ent_wiki_sm"):
            try:
                return spacy.load(model)
            except OSError:
                continue
        nlp_fallback = spacy.blank("xx")
        return nlp_fallback

    def _ensure_nltk(self) -> None:
        try:
            _ = stopwords.words('english')
        except LookupError:
            print("Downloading NLTK 'stopwords'...")
            nltk.download('stopwords')

        try:
            word_tokenize("test")
        except LookupError:
            print("Downloading NLTK 'punkt' tokenizer data...")
            nltk.download('punkt')

        try:
            nltk.data.find('tokenizers/punkt_tab')
        except LookupError:
            print("Downloading NLTK 'punkt_tab'...")
            nltk.download('punkt_tab')

        try:
            nltk.data.find('tokenizers/punkt/english')
        except LookupError:
            print("Downloading specific NLTK 'punkt' English tokenizer data...")
            try:
                nltk.download('punkt')
            except Exception as e:
                print(f"Failed to download 'punkt': {e}")
                pass

    def clean_text(self, text: str) -> str:
        """
        Convert text to lowercase,
        remove URLs and special characters,
        remove extra whitespace,
        keep accented letters.
        """
        text = str(text).lower()
        text = re.sub(r"[^a-zA-ZÀ-ÿ0-9\s]", "", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def lemmatize_text(self, text: str) -> str:
        """
        Groups different forms of words so that they can be analyzed as single items,
        e.g., 'running' becomes 'run', 'ran' also becomes 'run'.
        Lemmatize text using spaCy.
        """
        doc = self.nlp(text)
        return " ".join(token.lemma_ if token.lemma_ else token.text for token in doc)

    def remove_stopwords(self, text: str) -> str:
        """
        Remove stopwords like (the, is, in) that don't carry much sentiment.
        """
        tokens = word_tokenize(text)
        sw = set(stopwords.words("english"))
        token = [t for t in tokens if t not in sw]
        return " ".join(token)

    def clean_data(self, data: pd.DataFrame):
        try:
            Cleaner = DataCleaning()
            data['clean_text'] = data['review'].apply(Cleaner.clean_text)
            data['lemma_text'] = data['clean_text'].apply(Cleaner.lemmatize_text)
            data['final_text'] = data['lemma_text'].apply(Cleaner.remove_stopwords)

            data['label'] = data['rating'].apply(lambda r: 0 if r in (1, 2) else (1 if r == 3 else 2))
            data = data[['review', 'final_text', 'label']]
            data.to_csv(Clean_data, index=False)
            logging.info("dataset has been successfully cleaned")

            print(data.head())
            return data
        except Exception as e:
            logging.error(f"error occurred while cleaning the data {e}")



import warnings
import pickle as pk
import logging
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from nltk.probability import FreqDist
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from nltk import ne_chunk
from nltk import pos_tag
from nltk.corpus import wordnet
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
import re
import unicodedata
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
sastrawi_stemmer = StemmerFactory().create_stemmer()


warnings.filterwarnings("ignore")
logging.basicConfig(
    format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


'''UTILS FUNCTION'''


def word_count_func(text: int):
    '''
    Counts words within a string
    Args:
        text (str): String to which the function is to be applied, string
    Returns:
        Number of words within a string, integer
    '''
    return len(text.split())


'''END UTILS FUNCTION'''


'''TEXT CLEANSING FUNCTIONS'''


def remove_html_tags_func(text: str):
    '''
    Removes HTML-Tags from a string, if present
    Args:
        text (str): String to which the function is to be applied, string
    Returns:
        Clean string without HTML-Tags
    '''
    return BeautifulSoup(text, 'html.parser').get_text()


def remove_url_func(text: str):
    '''
    Removes URL addresses from a string, if present
    Args:
        text (str): String to which the function is to be applied, string
    Returns:
        Clean string without URL addresses
    '''
    return re.sub(r'https?://\S+|www\.\S+', '', text)


def remove_accented_chars_func(text: str):
    '''
    Removes all accented characters from a string, if present
    Args:
        text (str): String to which the function is to be applied, string
    Returns:
        Clean string without accented characters
    '''
    return unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')


def remove_punctuation_func(text: str):
    '''
    Removes all punctuation from a string, if present
    Args:
        text (str): String to which the function is to be applied, string
    Returns:
        Clean string without punctuations
    '''
    return re.sub(r'[^a-zA-Z0-9]', ' ', text)


def remove_irr_char_func(text: str):
    '''
    Removes all irrelevant characters (numbers and punctuation) from a string, if present
    Args:
        text (str): String to which the function is to be applied, string
    Returns:
        Clean string without irrelevant characters
    '''
    return re.sub(r'[^a-zA-Z]', ' ', text)


def remove_extra_whitespaces_func(text: str):
    '''
    Removes extra whitespaces from a string, if present
    Args:
        text (str): String to which the function is to be applied, string
    Returns:
        Clean string without extra whitespaces
    '''
    return re.sub(r'^\s*|\s\s*', ' ', text).strip()


'''END TEXT CLEANSING FUNCTIONS'''


'''STOPWORDS FUNCTIONS'''


def remove_english_stopwords_func(text: str):
    '''
    Removes Stop Words (also capitalized) from a string, if present
    Args:
        text (str): String to which the function is to be applied, string
    Returns:
        Clean string without Stop Words
    '''
    # check in lowercase
    t = [token for token in text if token.lower() not in stopwords.words("english")]
    text = ' '.join(t)
    return text


def remove_indo_stopwords_func(words):
    '''
    Removes Stop Words indoensian
    Args:
        text (str): String to which the function is to be applied, string
    Returns:
        Clean string without Stop Words
    '''
    list_stopwords = stopwords.words('indonesian')
    list_stopwords = set(list_stopwords)
    t = [word for word in words if word not in list_stopwords]
    text = ' '.join(t)
    return text


'''END STOPWORDS FUNCTIONS'''


'''POS, NER, NORMALIZATION FUNCTIONS'''


def norm_lemm_v_a_func(text):
    '''
    Lemmatize tokens from string

    Step 1: Use word_tokenize() to get tokens from string
    Step 2: Use WordNetLemmatizer() with POS tag 'v' to lemmatize the created tokens
    Step 3: Use word_tokenize() to get tokens from generated string
    Step 4: Use WordNetLemmatizer() with POS tag 'a' to lemmatize the created tokens

    Args:
        text (str): String to which the functions are to be applied, string

    Returns:
        String with lemmatized words
    '''
    words1 = word_tokenize(text)
    text1 = ' '.join([WordNetLemmatizer().lemmatize(word, pos='v')
                     for word in words1])
    words2 = word_tokenize(text1)
    text2 = ' '.join([WordNetLemmatizer().lemmatize(word, pos='a')
                     for word in words2])
    return text2


def norm_stemm_indo_func(text):
    words = word_tokenize(text)
    text1 = ' '.join([sastrawi_stemmer.stem(word) for word in words])
    return text1


'''END POS, NER, NORMALIZATION FUNCTIONS'''


'''REMOVING SINGLE CHAR FUNCTIONS'''


def remove_single_char_func(text, threshold=1):
    '''
    Removes single characters from string, if present

    Step 1: Use word_tokenize() to get tokens from string
    Step 2: Removes words whose length falls below the threshold (by default = 1)

    Args:
        text (str): String to which the functions are to be applied, string

    Returns:
        String with removed words whose length was below the threshold (by default = 1)
    '''
    threshold = threshold

    words = word_tokenize(text)
    text = ' '.join([word for word in words if len(word) > threshold])
    return text


'''END REMOVING SINGLE CHAR FUNCTIONS'''


'''REMOVE UNUSED WORDS FUNCTIONS'''


def single_word_remove_func(text, word_2_remove):
    '''
    Removes a specific word from string, if present

    Step 1: Use word_tokenize() to get tokens from string
    Step 2: Removes the defined word from the created tokens

    Args:
        text (str): String to which the functions are to be applied, string
        word_2_remove (str): Word to be removed from the text, string

    Returns:
        String with removed words
    '''
    word_to_remove = word_2_remove

    words = word_tokenize(text)
    text = ' '.join([word for word in words if word != word_to_remove])
    return text


def multiple_word_remove_func(text, words_2_remove_list):
    '''
    Removes certain words from string, if present

    Step 1: Use word_tokenize() to get tokens from string
    Step 2: Removes the defined words from the created tokens

    Args:
        text (str): String to which the functions are to be applied, string
        words_2_remove_list (list): Words to be removed from the text, list of strings

    Returns:
        String with removed words
    '''
    words_to_remove_list = words_2_remove_list

    words = word_tokenize(text)
    text = ' '.join(
        [word for word in words if word not in words_to_remove_list])
    return text


'''END REMOVE UNUSED WORDS FUNCTIONS'''


'''
    MAIN PROCESS
    Main Process means running all preprocessing procedures sequentially
'''


def fit(dataset: pd.DataFrame, column_name: str, log=True, lang='id') -> pd.DataFrame:
    '''
    This function used to implement all preprocessing procedure, supported language are english and indonesia

    Args:
        dataset (pd.DataFrame): DataFrame
        column_name (str): column name used to preprocessing, 
        log (bool): if you want to show log process default True
        lang (str): setting language default id

    Returns:
        Dataframe

    Step 1: cleansing 
    Step 2: tokenizing
    Step 3: removing stopwords
    Step 4: applying normalization (stemming/lemmatizing)
    Step 5: removing single char
    Step 6: removing unrelated words


    '''

    # assign loaded data
    df = dataset

    # apply cleansing
    (log and logging.info("Cleansing"))
    data = df[column_name]
    data = data.str.lower()
    data = data.apply(remove_html_tags_func)
    data = data.apply(remove_url_func)
    data = data.apply(remove_accented_chars_func)
    data = data.apply(remove_punctuation_func)
    data = data.apply(remove_irr_char_func)
    data = data.apply(remove_extra_whitespaces_func)
    df['CleanWord'] = data
    df['CleanWordCount'] = data.apply(word_count_func)

    # apply tokenizing
    (log and logging.info("Tokenizing"))
    df['Tokenized'] = df['CleanWord'].apply(word_tokenize)
    df['TokenCount'] = df['Tokenized'].str.len()

    # apply stop words
    (log and logging.info("Removing Stopwords"))

    df['StopWordsLess'] = df['Tokenized'].apply(remove_indo_stopwords_func)
    df['StopWordsLessCount'] = df['StopWordsLess'].apply(word_count_func)

    # apply normalizing
    (log and logging.info("normalizing"))
    import swifter
    df['Normalized'] = df['StopWordsLess'].swifter.set_npartitions(
        npartitions=4).apply(norm_stemm_indo_func)
    df['NormalizedCount'] = df['Normalized'].apply(word_count_func)

    # apply remove single char
    (log and logging.info("Removing single char"))
    df['SingleCharLess'] = df['StopWordsLess'].apply(remove_single_char_func)
    df['SingleCharLessCount'] = df['SingleCharLess'].apply(word_count_func)

    # apply remove unused words
    (log and logging.info("Remove unused words"))
    unused_words = []
    df['UnusedWordsLess'] = df.apply(
        lambda it: multiple_word_remove_func(it['SingleCharLess'], unused_words), axis=1)
    df['UnusedWordsLessCount'] = df['UnusedWordsLess'].apply(word_count_func)

    # assign last process to final result column
    df['Result'] = df['UnusedWordsLess']

    return df

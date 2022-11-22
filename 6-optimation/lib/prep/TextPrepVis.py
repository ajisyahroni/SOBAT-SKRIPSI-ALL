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

import pickle as pk

import warnings
warnings.filterwarnings("ignore")


'''FUNCTIONS'''


def token_and_unique_word_count_func(text):
    '''
    Outputs the number of words and unique words

    Step 1: Use word_tokenize() to get tokens from string
    Step 2: Uses the FreqDist function to count unique words

    Args:
        text (str): String to which the functions are to be applied, string

    Prints:
        Number of existing tokens and number of unique words
    '''
    words = word_tokenize(text)
    fdist = FreqDist(words)

    print('Number of tokens: ' + str(len(words)))
    print('Number of unique words: ' + str(len(fdist)))


def most_common_word_func(text, n_words=25):
    '''
    Returns a DataFrame with the most commonly used words from a text with their frequencies

    Step 1: Use word_tokenize() to get tokens from string
    Step 2: Uses the FreqDist function to determine the word frequency

    Args:
        text (str): String to which the functions are to be applied, string

    Returns:
        A DataFrame with the most commonly occurring words (by default = 25) with their frequencies
    '''
    words = word_tokenize(text)
    fdist = FreqDist(words)

    n_words = n_words

    df_fdist = pd.DataFrame({'Word': fdist.keys(),
                             'Frequency': fdist.values()})
    df_fdist = df_fdist.sort_values(
        by='Frequency', ascending=False).head(n_words)

    return df_fdist


def least_common_word_func(text, n_words=25):
    '''
    Returns a DataFrame with the least commonly used words from a text with their frequencies

    Step 1: Use word_tokenize() to get tokens from string
    Step 2: Uses the FreqDist function to determine the word frequency

    Args:
        text (str): String to which the functions are to be applied, string

    Returns:
        A DataFrame with the least commonly occurring words (by default = 25) with their frequencies
    '''
    words = word_tokenize(text)
    fdist = FreqDist(words)

    n_words = n_words

    df_fdist = pd.DataFrame({'Word': fdist.keys(),
                             'Frequency': fdist.values()})
    df_fdist = df_fdist.sort_values(
        by='Frequency', ascending=False).tail(n_words)

    return df_fdist


def least_common_word_func(text, n_words=25):
    '''
    Returns a DataFrame with the least commonly used words from a text with their frequencies

    Step 1: Use word_tokenize() to get tokens from string
    Step 2: Uses the FreqDist function to determine the word frequency

    Args:
        text (str): String to which the functions are to be applied, string

    Returns:
        A DataFrame with the least commonly occurring words (by default = 25) with their frequencies
    '''
    words = word_tokenize(text)
    fdist = FreqDist(words)

    n_words = n_words

    df_fdist = pd.DataFrame({'Word': fdist.keys(),
                             'Frequency': fdist.values()})
    df_fdist = df_fdist.sort_values(
        by='Frequency', ascending=False).tail(n_words)

    return df_fdist


'''END FUNCTIONS'''


'''
    MAIN PROCESS
    Main Process means running all preprocessing procedures sequentially
'''


def show(prep_res: pd.DataFrame) -> None:
    df = prep_res

    # convert last processes column to corpus
    text_corpus = df['Result'].str.cat(sep=' ')

    # visualizing as wc
    wordcloud = WordCloud(width=800, height=800,
                          background_color='white',
                          min_font_size=10).generate(text_corpus)
    plt.figure(figsize=(8, 8), facecolor=None)
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.show()

    # apply unique words
    token_and_unique_word_count_func(text_corpus)

    # apply most common words
    df_most_common_words_text_corpus = most_common_word_func(text_corpus)
    print(df_most_common_words_text_corpus.head(10))

    # visualizing most common words
    plt.figure(figsize=(11, 7))
    plt.bar(df_most_common_words_text_corpus['Word'],
            df_most_common_words_text_corpus['Frequency'])
    plt.xticks(rotation=45)
    plt.xlabel('Most common Words')
    plt.ylabel("Frequency")
    plt.title("Frequency distribution of the 25 most common words")
    plt.show()

    # apply least common words
    df_least_common_words_text_corpus = least_common_word_func(
        text_corpus, n_words=10)
    df_least_common_words_text_corpus
    plt.figure(figsize=(11, 7))
    plt.bar(df_least_common_words_text_corpus['Word'],
            df_least_common_words_text_corpus['Frequency'])
    plt.xticks(rotation=45)
    plt.xlabel('Least common Words')
    plt.ylabel("Frequency")
    plt.title("Frequency distribution of the 10 least common words")
    plt.show()

    # apply reduction info each process
    df['ReductionValue'] = round((
        df['CleanWordCount'] - df['UnusedWordsLessCount']) / df['CleanWordCount'] * 100)
    df_subset = df[['UnusedWordsLess', 'CleanWordCount', 'TokenCount',
                    'StopWordsLessCount', 'NormalizedCount',
                    'SingleCharLessCount', 'UnusedWordsLessCount', 'ReductionValue']]

    df_subset = df_subset.sort_values(by=['ReductionValue'])
    reduction_mean = df_subset['ReductionValue'].mean()
    print(f'reduction mean : {reduction_mean}%')
    # df_subset.to_csv('res/2019_06of50_train_labeled_prep_info.csv')

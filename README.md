# MEDICARE - Drug Recommender System

**Objective:**


>The objective of this project is to develop a drug recommendation system for medical practitioners to get information on the popular drugs in the market at any point of time.

**Functionality and Usage:**

> The main functionality of this tool is to recommend drugs available in the market to doctors based on a recommended rating score provided by patient’s reviews and medical conditions. 
> We intend to create this tool to help doctors get an idea about different drugs, with respect to the condition, based on reviews and ratings given by the patients. This would help them to shortlist the best drugs for each ailment and prescribe them accordingly. 
>This tool potentially eliminates the role of an intermediate medical representative from different pharmaceutical companies for briefing and awareness about the available drugs in the market.

**Dataset:**

>http://archive.ics.uci.edu/ml/datasets/Drug+Review+Dataset+%28Drugs.com%29


**Setting up the Environment:**

Libraries to install and import:

  
  import numpy as np
  
  import pandas as pd
  
  import os
  
  import spacy
  
  import en_core_web_sm
  
  from spacy.lang.en import English
  
  from spacy.lang.en.stop_words import STOP_WORDS
  
  import string
  
  from sklearn.model_selection import train_test_split
  
  from sklearn.linear_model import LogisticRegression
  
  from sklearn import metrics
  
  from sklearn.feature_extraction import DictVectorizer
  
  from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer,HashingVectorizer
  
  from sklearn.base import TransformerMixin
  
  from sklearn.pipeline import Pipeline
  
  import vaderSentiment
  
  from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
  
  import seaborn as sns
  
  import matplotlib.pyplot as plt
  
  from wordcloud import WordCloud
  
  import re
  
  from bs4 import BeautifulSoup
  
  import nltk
  
  nltk.download('stopwords')
  
  from nltk.corpus import stopwords
  
  from nltk.stem.snowball import SnowballStemmer
  
  from nltk.stem.porter import PorterStemmer
  
  import json
  
  import pickle
  
  import pandas as pd
  
  from sklearn.model_selection import train_test_split
  
  from sklearn.feature_extraction.text import CountVectorizer
  
  from sklearn.pipeline import Pipeline
  
  from sklearn.linear_model import LogisticRegression
  
  from sklearn.naive_bayes import MultinomialNB
  
  from sklearn.neighbors import KNeighborsClassifier
  
  from sklearn.svm import SVC
  
  from sklearn.ensemble import RandomForestClassifier
  
  from sklearn.ensemble import AdaBoostClassifier
  
  from sklearn.linear_model import SGDClassifier
  
  from sklearn.pipeline import Pipeline
  
  from sklearn.metrics import accuracy_score
  
  from mlxtend.classifier import StackingClassifier
  
  from sklearn.model_selection import cross_val_score, train_test_split
  
  nltk.download('vader_lexicon')

  import numpy as np
  
  from nltk.sentiment.vader import SentimentIntensityAnalyzer
  
 >**Methodology:**
  
   **Pre-Processing:**
  
        1.    Creating the token object, which is used to create documents with linguistic annotations
   
        2.    Lemmatizing each token and converting each token into lowercase
   
        3.    Removing stop words
   
        4.    Return preprocessed list of tokens


  
  

# MEDICARE - Drug Recommender System

**Objective:**


-The objective of this project is to develop a drug recommendation system for medical practitioners to get information on the popular drugs in the market at any point of time.

**Functionality and Usage:**

-The main functionality of this tool is to recommend drugs available in the market to doctors based on a recommended rating score provided by patient’s reviews and medical conditions.<br/>
-We intend to create this tool to help doctors get an idea about different drugs, with respect to the condition, based on reviews and ratings given by the patients. This would help them to shortlist the best drugs for each ailment and prescribe them accordingly.<br/> 
-This tool potentially eliminates the role of an intermediate medical representative from different pharmaceutical companies for briefing and awareness about the available drugs in the market.

![functionality](https://user-images.githubusercontent.com/54454914/81767023-68fac100-94a5-11ea-9426-4ddbf63f1e1f.jpg)


**Dataset:**

http://archive.ics.uci.edu/ml/datasets/Drug+Review+Dataset+%28Drugs.com%29


**Setting up the Environment:**

The below libraries need to be installed and imported using Python3.0 or higher:

  
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
  
 **Methodology:**
 
 **Text Preprocessing** 
 
    Text Pre-Processing using Snowball Stemmer:
        
        1.    Delete HTML 
        2.    Removing special characters and keeping only letters
        3.    Convert to lower-case
        4.    Removing Stop words
        5.    Stemming
        6.    Joining stemming words

    Text Pre-Processing using Spacy Tokenizer:
  
        1.    Creating the token object, which is used to create documents with linguistic annotations
        2.    Lemmatizing each token and converting each token into lowercase
        3.    Removing stop words   
        4.    Return preprocessed list of tokens
        
  **Sentiment Analysis:**
  
   - Vader Sentiment Analysis was performed on the preprocessed reviews.<br/>
   - Categorizes reviews into Positive(>=0.05), Neutral(between -0.05& 0.05) and Negative(<=-0.05)​
      
  **Machine Learning Models used for verifying score:**
  
   3 Machine Learning models were used :
  
   * Logistic Regression
   * SGD
   * Multinomial Naive Bayes
   
   Logistic Regression had the maximum accuracy.
   
  **Normalization and recommended mean score:**
  
        1.   Normalized the review rating to match the 0-10 rating scale
        2.   Final drug rating was calculated as mean of the rating and normalized vader review score.
        3.   Sorted the data based on Condition and Drug name and aggregate was taken for the normalized rating. 
        4.   Recommendation was done by ranking the sorted data.
      
  **Dashboard Creation:**
  
        1.    Used Tableau Desktop/Public for Dashboard Creation
        2.    Embedded the Dashboard into a static website
        3.    Hosted the website on GitHub
    
   ![DASHBOARD_MEDICARE](https://user-images.githubusercontent.com/54454914/81767095-90518e00-94a5-11ea-899d-688a72c8c62d.jpg)
        
   Link to the dashboard : https://ldsouza13.github.io/MEDICARE.github.io/
  
  **Accomplishments**
  
This MEDICARE application developed based on a normalized recommended score, will help the doctors to view the top rated drugs for a    particular condition/disease by using a user-friendly and interactive web application. 

  **Future Scope**
  
-Although we tried removing spam reviews while preprocessing the data, the dataset had too many duplicate reviews. A proper spam removal model could be developed for the same in the future.<br/>
-We tried implementing a recommendation system using content based and collaborative item-based filtering. However, the huge data set led to sparse matrix and inaccurate results. A hybrid system could be used in the future. 
(The item based Collaborative Filtering code is uploaded)

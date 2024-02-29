import spacy

import streamlit as st

from scipy.spatial import distance

from sklearn.feature_extraction.text import TfidfVectorizer

import re 

from googletrans import Translator

from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords

import warnings
warnings.filterwarnings("ignore")

import subprocess

# @st.cache_resource
# def download_en_core_web_sm():
#     subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])



tfidf_vectorizer = TfidfVectorizer()
translator = Translator()


def prepare_corpus(corpus):
    translation = translator.translate(corpus, dest='en')
    trans_corpus = translation.text
    return trans_corpus

def prepare_links(link_text):
    new_txt_list = []

    text_2 = re.split(r'(?=[.]\s[А-ЯA-Z])', link_text)
    for i in text_2:
        new_txt_list.append(re.sub(r'([.]\s)', '', i))

    return new_txt_list


def prework_general(text):
    # try:
    #     nlp = spacy.load('en_core_web_sm')
    
    # except:
    #     subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
    #     nlp = spacy.load('en_core_web_sm')
    nlp = spacy.load('en_core_web_sm')

    tokenizer = RegexpTokenizer(r'[а-яА-Яa-zA-Z0-9]+\-?[а-яА-Яa-zA-Z0-9]+')

    tokens = [token for token in tokenizer.tokenize(text.lower())]
    
    filtered_tokens = " ".join([word for word in tokens if not word in stopwords.words('english')])
    
    doc = nlp(filtered_tokens) 

    documents = [' '.join([token.lemma_ for token in doc])]

    return documents

def prework(text_1, text_2):
    link_sentence = []

    corpus = prework_general(text_1)

    for sentence in text_2:
        link_sentence.append(prework_general(sentence))
    
    return corpus, link_sentence


def tf_idf(corpus, links):
    tfidf_links = []

    tfidf_corpus = tfidf_vectorizer.fit_transform(corpus)
    vector_corpus = tfidf_corpus.toarray()[0]

    for sentence in links:
        vector_link = tfidf_vectorizer.transform(sentence)
        tfidf_links.append(vector_link.toarray()[0])
            
    return vector_corpus, tfidf_links

    

def main(text_1, text_2):
    t_1 = prepare_corpus(text_1)
    t_2 = prepare_links(text_2)
    
    corpus, links = prework(t_1, t_2)

    st.title('Sentences from link')
    for i in range(len(t_2)):
        st.write(f'({i}) --> {t_2[i]}')
        

    tf_idf_corpus, tf_idf_links = tf_idf(corpus, links)

    st.title('Answers')

    for j in range(len(tf_idf_links)):
        dist = 1 - distance.cosine(tf_idf_corpus, tf_idf_links[j])
        if dist >= 0.3:
            st.success(f'({j}) --> cos_distance: {dist}')
            st.success(f'{t_2[j]}')

        else:
            st.error(f'({j}) --> cos_distance: {dist}')
            st.error(f'{t_2[j]}')

st.title('App for validation links')
corpus = st.text_area('Text from article')
links = st.text_area('Text from link')  

if st.button('Tap to submit'):
    main(corpus, links)


# починить слова с - lithium-ion ==> lithium - ion
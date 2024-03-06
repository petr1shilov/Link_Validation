import spacy

import pandas as pd 

import streamlit as st

from scipy.spatial import distance

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

import re 

from googletrans import Translator

import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords

import warnings
warnings.filterwarnings("ignore")

nltk.download('stopwords')

tfidf_vectorizer = TfidfVectorizer()
vector_wb = CountVectorizer()
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
    nlp = spacy.load('en_core_web_sm')

    tokenizer = RegexpTokenizer(r'[а-яА-Яa-zA-Z0-9]+\-?[а-яА-Яa-zA-Z0-9]+')

    tokens = [token for token in tokenizer.tokenize(text.lower())]
    
    filtered_tokens = " ".join([word for word in tokens if not word in stopwords.words('english')])
    
    doc = nlp(filtered_tokens) 

    documents = ' '.join([token.lemma_ for token in doc])

    return documents

def prework(text_1, text_2):
    link_sentence = []

    query_text = prework_general(text_1)

    for sentence in text_2:
        link_sentence.append(prework_general(sentence))

    corpus = list(link_sentence)
    corpus.append(query_text)
    query_text = [query_text]
    
    return query_text, link_sentence, corpus

def counter_vec(query, links, corpus):
    vector_wb.fit(corpus)
    count_query = vector_wb.transform(query).toarray()
    count_links = vector_wb.transform(links).toarray()

    return count_query, count_links

def tf_idf(query, links, corpus):
    tfidf_vectorizer.fit(corpus)
    vector_query = tfidf_vectorizer.transform(query).toarray()
    tfidf_links = tfidf_vectorizer.transform(links).toarray()
            
    return vector_query, tfidf_links 

def main(text_1, text_2):
    t_1 = prepare_corpus(text_1)
    t_2 = prepare_links(text_2)

    st.title('Sentences from link')
    for i in range(len(t_2)):
        st.write(f'({i}) --> {t_2[i]}')
    
    query, links, corpus = prework(t_1, t_2)

    counter_query, counter_links = counter_vec(query, links, corpus)
    tf_idf_query, tf_idf_links = tf_idf(query, links, corpus)

    counter = []
    tf_idf_count = []
    for j in range(len(counter_links)):
        dist_1 = 1 - distance.cosine(counter_query[0], counter_links[j]) # cos_dist, 1 - cos_dist = cos_sim 
        counter.append(dist_1)
        dist_2 = 1 - distance.cosine(tf_idf_query[0], tf_idf_links[j]) # cos_dist, 1 - cos_dist = cos_sim 
        tf_idf_count.append(dist_2)

    df = pd.DataFrame({'cos_similarity_counter' : counter, 'cos_similarity_tf_idf' : tf_idf_count}) #'text' : t_2,
    df['mean'] = (df.cos_similarity_counter + df.cos_similarity_tf_idf) / 2
    # st.markdown('<style>div[title="OK"] { color: green; } div[title="KO"] { color: red; } .data:hover{ background:rgb(243 246 255)}</style>', unsafe_allow_html=True)
    # st.markdown('<style>div[mean>0.1] { color: green; }  .df:hover{ background:rgb(243 246 255)}</style>', unsafe_allow_html=True)

    st.title('Answers')

    for j in range(len(tf_idf_links)):
        dist_tf_idf = 1 - distance.cosine(tf_idf_query[0], tf_idf_links[j])
        dist_counter = 1 - distance.cosine(counter_query[0], counter_links[j])
        if dist_counter >= 0.11:
            st.success(f'({j}) --> cos_similarity_counter: {dist_counter} and cos_similarity_tf_idf: {dist_tf_idf}')
            st.success(f'{t_2[j]}')

        else:
            st.error(f'({j}) --> cos_similarity_counter: {dist_counter} and cos_similarity_tf_idf: {dist_tf_idf}')
            st.error(f'{t_2[j]}')    
    
    st.dataframe(df)

st.title('App for validation links')
query = st.text_area('Text from article')
links = st.text_area('Text from link')  

if st.button('Tap to submit'):
    main(query, links)



    
# text_1 = 'Такие аккумуляторы привлекательны с маркетинговой точки зрения, в том числе из-за низкой стоимости серы по сравнению с кобальтом, используемым в существующих литий-ионных аккумуляторах'
# text_2 = 'Safe, low-cost, high-energy-density and long-lasting recharge- able batteries are in high demand to address pressing environmental needs for energy storage systems that can be coupled to renewable sources1,2. These include wind, wave and solar energy, as well as regenerative braking from vehicular transport. With production of oil predicted to decline, and the number of vehicles and their pollution impact increasing globally, a transformation in transportation economy is inevitable given that we live in a carbon-constrained world. One of the most promising candidates for storage devices is the lithium–sulphur cell. Under intense scrutiny for well over two decades, the cell in its simplest configuration consists of sulphur as the positive electrode and lithium as the negative electrode3,4. It differs from conventional lithium-ion cells, which operate on the basis of topotactic inter- calation reactions: reversible uptake of Li ions and electrons in a solid with minimal change to the structure.'
# main(text_1, text_2)


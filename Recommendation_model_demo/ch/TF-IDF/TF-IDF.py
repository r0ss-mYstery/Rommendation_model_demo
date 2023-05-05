from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from gensim.models import KeyedVectors
from scipy.spatial.distance import cityblock
import pandas as pd
import numpy as np
import jieba

cleaned_data_path = "../data/title/cleaned_titles_ch5.csv"
df = pd.read_csv(cleaned_data_path)

def custom_tokenizer(tokens):
    return tokens


df['cleaned_title'] = df['cleaned_title'].fillna('')
df['tokens'] = df['cleaned_title'].apply(lambda x: x.split())
vectorizer = TfidfVectorizer(tokenizer=custom_tokenizer, lowercase=False, preprocessor=lambda x: x)
tfidf_matrix = vectorizer.fit_transform(df['tokens'])

def title_to_tokens(title):
    tokens = list(jieba.cut(title))
    return tokens


def get_top_n_similar_titles(input_title, df, n=5):
    input_tokens = title_to_tokens(input_title)
    query_vector = vectorizer.transform([input_tokens])
    similarity_scores = cosine_similarity(query_vector, tfidf_matrix)
    sorted_indices = np.argsort(-similarity_scores[0])
    top_n_indices = sorted_indices[1:n + 1]

    top_n_titles_list = []
    for idx in top_n_indices:
        title_id = df.iloc[idx]['product_id']  # Update this line with the correct ID column name
        title_text = df.iloc[idx]['product_title']
        top_n_titles_list.append({"id": title_id, "title": title_text})
    top_n_titles_df = pd.DataFrame(top_n_titles_list)

    return top_n_titles_df


input_title = "德州扑克桌"
top_n_titles_df = get_top_n_similar_titles(input_title, df, n=20)
print(top_n_titles_df)


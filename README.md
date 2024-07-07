# Movie Recommender System

This project is a movie recommendation system that suggests movies based on a selected movie using content-based filtering. The system uses movie metadata and provides recommendations through a web interface built with Streamlit.

# Table of Contents
# Installation
# Usage
# How It Works
# File Descriptions
# Credits
# Installation
# Prerequisites
# Python 3.6+
# pip (Python package installer)
# Libraries
# Install the required libraries using pip:

bash

Copy code

pip install numpy pandas scikit-learn nltk streamlit requests

Data Files

Ensure you have the following CSV files in your working directory:

tmdb_5000_movies.csv
tmdb_5000_credits.csv
Additional Files
Ensure you have the background images in your working directory:

photo.jpg (light mode background image)
dark theam.jpg (dark mode background image)
Usage
Prepare the Data:

# Run the script to preprocess the data and create necessary pickle files:

python
Copy code
import numpy as np
import pandas as pd
import ast
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.stem.porter import PorterStemmer
import pickle

# Load datasets
movies = pd.read_csv(r'tmdb_5000_movies.csv')
credits = pd.read_csv(r'tmdb_5000_credits.csv')

# Merge datasets
movies = movies.merge(credits, on='title')
movies = movies[['movie_id', 'title', 'overview', 'genres', 'keywords', 'cast', 'crew']]

# Define helper functions
def convert(text):
    L = []
    for i in ast.literal_eval(text):
        L.append(i['name']) 
    return L 

def convert3(text):
    L = []
    counter = 0
    for i in ast.literal_eval(text):
        if counter < 3:
            L.append(i['name'])
        counter += 1
    return L 

def fetch_director(text):
    L = []
    for i in ast.literal_eval(text):
        if i['job'] == 'Director':
            L.append(i['name'])
    return L 

def collapse(L):
    L1 = []
    for i in L:
        L1.append(i.replace(" ", ""))
    return L1

# Preprocess data
movies.dropna(inplace=True)
movies['genres'] = movies['genres'].apply(convert)
movies['keywords'] = movies['keywords'].apply(convert)
movies['cast'] = movies['cast'].apply(convert3)
movies['cast'] = movies['cast'].apply(lambda x: x[0:3])
movies['crew'] = movies['crew'].apply(fetch_director)
movies['overview'] = movies['overview'].apply(lambda x: x.split())
movies['cast'] = movies['cast'].apply(collapse)
movies['crew'] = movies['crew'].apply(collapse)
movies['genres'] = movies['genres'].apply(collapse)
movies['keywords'] = movies['keywords'].apply(collapse)
movies['tags'] = movies['overview'] + movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew']

new = movies.drop(columns=['overview', 'genres', 'keywords', 'cast', 'crew'])
new['tags'] = new['tags'].apply(lambda x: " ".join(x))
new['tags'] = new['tags'].apply(lambda x: x.lower())
ps = PorterStemmer()

def stem(text):
    y = []
    for i in text.split():
        y.append(ps.stem(i))
    return " ".join(y)

new['tags'] = new['tags'].apply(stem)

# Save processed data
pickle.dump(new.to_dict(), open('movie_dict.pkl', 'wb'))
movie_dict = pickle.load(open('movie_dict.pkl', 'rb'))

# Vectorize tags
cv = CountVectorizer(max_features=5000, stop_words='english')
vector = cv.fit_transform(new['tags']).toarray()

# Compute similarity
similarity = cosine_similarity(vector)
pickle.dump(similarity, open('similarity.pkl', 'wb'))
Run the Streamlit App:

Save the following Streamlit app code in a file named app.py:

python
Copy code
import streamlit as st
import pickle
import pandas as pd
import requests
import base64

def fetch_poster(movie_id):
    response = requests.get(f'https://api.themoviedb.org/3/movie/{movie_id}?api_key=30186192052c476dd889028457fc0d66')
    data = response.json()
    return "http://image.tmdb.org/t/p/w500/" + data['poster_path']

def recommend(movie, num_recommendations=10):
    movie_index = movies[movies['title'] == movie].index[0]
    distances = similarity[movie_index]
    movie_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:num_recommendations+1]
    
    recommend_movies = []
    recommend_movies_poster = []
    for i in movie_list:
        movie_id = movies.iloc[i[0]].movie_id
        recommend_movies.append(movies.iloc[i[0]].title)
        recommend_movies_poster.append(fetch_poster(movie_id))
    
    return recommend_movies, recommend_movies_poster

# Load the movie dictionary from the pickle file
with open('movie_dict.pkl', 'rb') as file:
    movies_dict = pickle.load(file)

# Create a DataFrame from the movie dictionary
movies = pd.DataFrame(movies_dict)

# Load the similarity matrix from the pickle file
with open('similarity.pkl', 'rb') as file:
    similarity = pickle.load(file)

# Function to convert image to base64
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

# Add background images for light and dark modes
img_light_path = 'photo.jpg'  # Replace with your light mode image file path
img_dark_path = 'dark theam.jpg'    # Replace with your dark mode image file path
img_light_base64 = get_base64_of_bin_file(img_light_path)
img_dark_base64 = get_base64_of_bin_file(img_dark_path)

# Toggle for dark mode
dark_mode = st.checkbox('Dark Mode')

# Set background image based on dark mode toggle
if dark_mode:
    img_base64 = img_dark_base64
else:
    img_base64 = img_light_base64

st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url("data:image/jpg;base64,{img_base64}");
        background-size: cover;
    }}
    .title {{
        color: red;
        font-size: 3em;
        font-weight: bold;
    }}
    .recommend-button {{
        background-color: skyblue;
        color: black;
        border: none;
        padding: 10px 20px;
        text-align: center;
        text-decoration: none;
        display: inline-block;
        font-size: 16px;
        margin: 4px 2px;
        cursor: pointer;
        border-radius: 12px;
    }}
    .recommend-button:hover {{
        background-color: lightblue;
    }}
    .recommendation {{
        color: #00008B; /* Dark blue color */
    }}
    </style>
    """,
    unsafe_allow_html=True
)

# Set the title of the Streamlit app with custom CSS class
st.markdown('<h1 class="title">Movie Recommender System</h1>',unsafe_allow_html=True)

# Initialize session state variables
if 'num_recommendations' not in st.session_state:
    st.session_state.num_recommendations = 10

# Create a select box for movie selection
selected_movie_name = st.selectbox(
    'Select a movie:',
    movies['title'].values
)

# Create a button to get recommendations
if st.button('Recommend'):
    st.session_state.num_recommendations = 10
    names, posters = recommend(selected_movie_name, st.session_state.num_recommendations)
    
    # Display all recommended movies
    cols = st.columns(5)
    for idx, col in enumerate(cols):
        col.text(names[idx])
        col.image(posters[idx])
    
    # Create rows for the rest of the recommendations
    for i in range(5, len(names), 5):
        row = st.columns(5)
        for idx, col in enumerate(row):
            if i + idx < len(names):
                col.text(names[i + idx])
                col.image(posters[i + idx])
# Run the app:

bash
Copy code
streamlit run app.py
How It Works
The Movie Recommender System uses a content-based filtering approach to recommend movies. It processes metadata from the TMDB dataset, including genres, keywords, cast, and crew, to create a "tags" column. This column is vectorized, and cosine similarity

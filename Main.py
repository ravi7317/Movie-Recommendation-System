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

    

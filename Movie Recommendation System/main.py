
import pandas as pd
import numpy as np
import ast

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity



print("Loading movie dataset...")

import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

movies = pd.read_csv(os.path.join(BASE_DIR, "movies.csv"))

print("Dataset loaded successfully!")
print("Total movies:", len(movies))


movies = movies[
    ["title", "genres", "keywords", "overview"]
].copy()


movies.dropna(subset=["title"], inplace=True)

movies.fillna({
    "genres": "",
    "keywords": "",
    "overview": ""
}, inplace=True)

movies.drop_duplicates(subset=["title"], inplace=True)

movies.reset_index(drop=True, inplace=True)

def extract_names(text):

    try:
        data = ast.literal_eval(text)

        return " ".join(
            item["name"] for item in data
            if isinstance(item, dict) and "name" in item
        )

    except (ValueError, SyntaxError, TypeError):
        return ""


movies["genres"] = movies["genres"].apply(extract_names)

movies["keywords"] = movies["keywords"].apply(extract_names)

movies["tags"] = (
    movies["genres"] + " " +
    movies["keywords"] + " " +
    movies["overview"]
)

movies["tags"] = movies["tags"].str.lower()

print("\nMovie information prepared!")

print("\nConverting movie descriptions into numbers...")

tfidf = TfidfVectorizer(
    max_features=5000,
    stop_words="english"
)

movie_vectors = tfidf.fit_transform(movies["tags"])

print("Text conversion completed!")

def recommend_movies(movie_name):

    movie_name = movie_name.strip().lower()

    matches = movies[
        movies["title"].str.lower() == movie_name
    ]

    if matches.empty:

        print("\nMovie not found!")

        suggestions = movies[
            movies["title"].str.lower().str.contains(
                movie_name, regex=False, na=False
            )
        ]

        if not suggestions.empty:
            print("\nDid you mean:")

            for title in suggestions["title"].head(5):
                print("-", title)

        return

    movie_index = matches.index[0]

    # Calculate cosine similarity
    similarity_scores = cosine_similarity(
        movie_vectors[movie_index],
        movie_vectors
    ).flatten()

    similar_movies = np.argsort(similarity_scores)[::-1]


    count = 0

    for index in similar_movies:

        if index == movie_index:
            continue

        print(
            f"{count + 1}. {movies.iloc[index]['title']}"
        )

        count += 1

        if count == 5:
            break


while True:

    movie_name = input(
        "\nEnter a movie name (or type exit): "
    )

    if movie_name.lower().strip() == "exit":
        print("Thank you for using our system!")
        break

    if not movie_name.strip():
        print("Please enter a movie name.")
        continue

    recommend_movies(movie_name)

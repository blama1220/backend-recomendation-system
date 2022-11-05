import pandas as pd  # Pandas - Used for data analysis.
import numpy as np  # Numpy - Used for working with arrays.
# Matplotlib - Used for visual representation like plotting graphs.
import matplotlib.pyplot as plt
# Sklearn - Used for making use of Machine learning tools.
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
# Ast - This module helps python application to process trees of the python abstract syntax grammar.
from ast import literal_eval
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# I load my credis.csv dataset in df1 dataframe and view it.
df1 = pd.read_csv("./data/tmdb_5000_credits.csv")
df1.head(10)
df1.shape

# Check null values
print("Info: ")
df1.info()
df1.isnull().sum()

# I load my movie.csv dataset in df2 dataframe and view it.
df2 = pd.read_csv("./data/tmdb_5000_movies.csv")
df2.head()
# Check null values
df2.info()
df2.isnull().sum()

# Merging df1 and df2
df1.columns = ['id', 'title', 'cast', 'crew']
df2 = df2.merge(df1, on="id")

print("Df2 Head :")
df2.head()

# Mean vote across the whole report
C = df2['vote_average'].mean()

# Mininum votes to be listed = 90%
m = df2["vote_count"].quantile(0.9)

#getting the list of movies tobe listed
movies_list = df2.copy().loc[df2["vote_count"]>=m]
movies_list.shape

# Defining a function
def weighted_rating(x, m=m, C=C):
    V = x['note_count']
    R = x["vote_average"]
    # Calculation done using IMDB formula
    return (V/(V+m) * R) + (m/(m+V) * C)


# Defining a new feature "score" and calculating its value with weighted rating
movies_list["score"] = movies_list.apply(weighted_rating, axis=1)

movies_list.head()


# Sort the movies based on their score
movies_list = movies_list.sort_values("score", ascending=False)
movies_list[["title_x", "vote_count", "vote_average", "score"]].head(10)


# Most popular movies
popular = df2.sort_values("popularity", ascending=False)
plt.figure(figsize=(12, 4))

plt.barh(popular["title_x"].head(10),
         popular["popularity"].head(10), align="center", color="darkblue")
plt.gca().invert_yaxis()
plt.xlabel("popularity")
plt.title("popular movies")


# Sort the movies based on their budget
budget = df2.sort_values("budget",ascending= False)
plt.figure(figsize=(12, 4))

plt.barh(budget["title_x"].head(10),
         budget["budget"].head(10), align="center", color="lightblue")
plt.gca().invert_yaxis()
plt.xlabel("budget")
plt.title("High budget movies")

print("Head df2 overview")
df2['overview'].head(10)


#Defining a TF-IDF vectorizer object and removing all stop words
tfidf = TfidfVectorizer(stop_words= "english")

#replacing NaN with empty string
df2["overview"] = df2["overview"].fillna("")

#making the TF-IDF matrix by fitting and transforming the data
tfidf_matrix = tfidf.fit_transform(df2["overview"])

#output the shape ofthe tfidf_matrix
tfidf_matrix.shape

#Calculating the cosine similarity matrix
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

#Creating a reverse map of indices and movies titles
indices = pd.Series(df2.index, index = df2["title_x"]).drop_duplicates()

#Defining a function that taje in a movie title as input and ouputs mos similar movies
def get_recomendations(title, cosine_sim = cosine_sim):
    #index of the movie that matches the title
    idx = indices[title]

    #pairwise similarity scores of all movies with that movie
    sim_scores = list(enumerate(cosine_sim[idx]))

    #Sorting the movies based on the similarity scores
    sim_scores = sorted(sim_scores, key = lambda x: x[1],reverse = True)

    #Scores of the 10 most similar movies
    sim_scores = sim_scores[1:11]

    #Get the movie index
    movie_indices = [i[0] for i in sim_scores]

    #Return the top 10 most similar movies
    return df2["title_x"].iloc[movie_indices]

print("--------RESULT-------")

get_recomendations("Fight Club")
from re import A
from matplotlib.dates import YearLocator
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

# Load datasets 
df1 = pd.read_csv("./data/tmdb_5000_credits.csv")
df2 = pd.read_csv("./data/tmdb_5000_movies.csv")

# Merging df1 and df2
df1.columns = ['id', 'title', 'cast', 'crew']
df2 = df2.merge(df1, on='id')

# Mean vote across the whole report
c = df2['vote_average'].mean()

# Mininum votes to be listed = 90%
m = df2['vote_count'].quantile(0.90)

#getting the list of movies to be listed
movies_list = df2.copy().loc[df2['vote_count'] >= m]


#TODO leer https://en.wikipedia.org/wiki/Bayes_estimator
# Defining a function Bayesian estimate
def weighted_rating(x, m=m, c=c):
	v = x['vote_count']
	R = x['vote_average']
	# Calculation done using IMDB formula
	return (v/(v+m) * R) + (m/(m+v) * c)

movies_list['score'] = movies_list.apply(weighted_rating, axis=1)
#print(movies_list.head())
movies_list = movies_list.sort_values('score', ascending=False)
#print(movies_list[['title_x', 'vote_count', 'vote_average', 'score']].head(10))
tfidf = TfidfVectorizer(stop_words = 'english')

#replacing NaN with empty string
df2['overview'] = df2['overview'].fillna('')

#making the TF-IDF matrix by fitting and transforming the data
tfidf_matrix = tfidf.fit_transform(df2['overview'])
print("Shape")
print(tfidf_matrix.shape)
#Calculating the cosine similarity matrix
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

#Creating a reverse map of indices and movies titles
indices = pd.Series(df2.index, index=df2['title_x']).drop_duplicates()




#-------------------------------
# Most popular movies
popular = df2.sort_values("popularity", ascending=False)
plt.figure(figsize=(12, 4))

plt.barh(popular["title_x"].head(10),
         popular["popularity"].head(10), align="center", color="darkblue")
plt.gca().invert_yaxis()
plt.xlabel("popularity")
plt.title("popular movies")
#-------------------------------



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




print("-----RECOMENDED-----")
print("Matrix Recomendations:")
print("ID                      Title")
print(get_recomendations("The Matrix"))


print("-----RECOMENDED-----")
print("Titanic Recomendations:")
print("ID                      Title")
print(get_recomendations("Titanic"))

 
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

df_credits = pd.read_csv("./tmdb_5000_credits.csv")
df_movies = pd.read_csv("./tmdb_5000_movies.csv")

df_movies.head()
df_credits.head()

print(df_movies["overview"])

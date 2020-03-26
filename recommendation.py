#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 25 23:51:48 2020

@author: savio
"""

import pandas as pd
df = pd.read_csv("IMDB_5000.csv")

colss = ["actor_1_name","genres","cast_total_facebook_likes","imdb_score"]   #recommendation based on actor name,genre,cast likes and imdb score
                                                 #you can add more factors if you like
for col in colss: 
    df[col] = df[col].fillna(" ")                 #remove null values by spaces

    
def combine_feature(row):
    return row["actor_1_name"]+" " + row["genres"] + " " + str(row["cast_total_facebook_likes"]) + " " + str(row["imdb_score"])

df["combine_feature"] = df.apply(combine_feature,axis = 1)

df.iloc[0]["combine_feature"]



from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer()
countmatrix = cv.fit_transform(df["combine_feature"])
cosine = cosine_similarity(countmatrix)

cosine_model_df = pd.DataFrame(cosine,columns = df["movie_title"])

z = input("Enter the Movie you Like:  ")
z = z + '\xa0'    #had to append '\xa0' because the created column had this already appended and needed to be appended because user dosent know about this'

print(cosine_model_df[z].sort_values(ascending = False)[:10])


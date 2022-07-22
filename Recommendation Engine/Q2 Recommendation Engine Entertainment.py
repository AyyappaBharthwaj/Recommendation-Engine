# import os
import pandas as pd

# import Dataset 
Entertain = pd.read_csv("E:/Data Science 18012022/Recommendation Engine/Entertainment.csv", encoding = 'utf8')
Entertain.shape # shape
Entertain.columns
Entertain.Category # genre columns

from sklearn.feature_extraction.text import TfidfVectorizer #term frequencey- inverse document frequncy is a numerical statistic that is intended to reflect how important a word is to document in a collecion or corpus

# Creating a Tfidf Vectorizer to remove all stop words
tfidf = TfidfVectorizer(stop_words = "english")    # taking stop words from tfid vectorizer 

# replacing the NaN values in overview column with empty string
Entertain["Category"].isnull().sum() 
Entertain["Category"] = Entertain["Category"].fillna(" ")

# Preparing the Tfidf matrix by fitting and transforming
tfidf_matrix = tfidf.fit_transform(Entertain.Category)   #Transform a count matrix to a normalized tf or tf-idf representation
tfidf_matrix.shape #51, 34

# with the above matrix we need to find the similarity score
# There are several metrics for this such as the euclidean, 
# the Pearson and the cosine similarity scores

# For now we will be using cosine similarity matrix
# A numeric quantity to represent the similarity between 2 movies 
# Cosine similarity - metric is independent of magnitude and easy to calculate 

# cosine(x,y)= (x.y⊺)/(||x||.||y||)

from sklearn.metrics.pairwise import linear_kernel

# Computing the cosine similarity on Tfidf matrix
cosine_sim_matrix = linear_kernel(tfidf_matrix, tfidf_matrix)

# creating a mapping of anime name to index number 
Entertain_index = pd.Series(Entertain.index, index = Entertain['Titles']).drop_duplicates()

Entertain_id = Entertain_index["Sabrina (1995)"]
Entertain_id

def get_recommendations(Name, topN):    
    # topN = 10
    # Getting the movie index using its title 
    Entertain_id = Entertain_index[Name]
    
    # Getting the pair wise similarity score for all the anime's with that 
    # anime
    cosine_scores = list(enumerate(cosine_sim_matrix[Entertain_id]))
    
    # Sorting the cosine_similarity scores based on scores 
    cosine_scores = sorted(cosine_scores, key=lambda x:x[1], reverse = True)
    
    # Get the scores of top N most similar movies 
    cosine_scores_N = cosine_scores[0: topN+1]
    
    # Getting the movie index 
    Entertain_idx  =  [i[0] for i in cosine_scores_N]
    anime_scores =  [i[1] for i in cosine_scores_N]
    
    # Similar movies and scores
    ET_similar_show = pd.DataFrame(columns=["name", "Score"])
    ET_similar_show["name"] = Entertain.loc[Entertain_idx, "Titles"]
    ET_similar_show["Score"] = anime_scores
    ET_similar_show.reset_index(inplace = True)  
    # anime_similar_show.drop(["index"], axis=1, inplace=True)
    print (ET_similar_show)
    # return (anime_similar_show)

    
# Enter your anime and number of anime's to be recommended 
get_recommendations("Babe (1995)", topN = 10)
Entertain_index["Babe (1995)"]

##### Best Recommended movies to Babe 1995 are Get Shorty (1995) and  Pocahontas (1995)

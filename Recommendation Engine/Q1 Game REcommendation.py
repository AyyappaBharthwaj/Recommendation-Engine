# import os
import pandas as pd

# import Dataset 
Gaming = pd.read_csv("E:/Data Science 18012022/Recommendation Engine/game.csv", encoding = 'utf8')
Gaming1 = Gaming.drop_duplicates(subset='game', keep='first', inplace=False)
Gaming1.shape # shape
Gaming1.columns
Gaming1.game # genre columns


from sklearn.feature_extraction.text import TfidfVectorizer #term frequencey- inverse document frequncy is a numerical statistic that is intended to reflect how important a word is to document in a collecion or corpus

# Creating a Tfidf Vectorizer to remove all stop words
tfidf = TfidfVectorizer(stop_words = "english")    # taking stop words from tfid vectorizer 

# replacing the NaN values in overview column with empty string
Gaming1["game"].isnull().sum() 


# Preparing the Tfidf matrix by fitting and transforming
tfidf_matrix = tfidf.fit_transform(Gaming1.game)   #Transform a count matrix to a normalized tf or tf-idf representation
tfidf_matrix.shape #3438,2068

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
Gaming_index = pd.Series(Gaming1.index, index = Gaming1['game']).drop_duplicates()

Gaming_id = Gaming_index["Grand Theft Auto IV"]
Gaming_id

def get_recommendations(Name, topN):    
    # topN = 10
    # Getting the movie index using its title 
    Gaming_id = Gaming_index[Name]
    
    # Getting the pair wise similarity score for all the anime's with that 
    # anime
    cosine_scores = list(enumerate(cosine_sim_matrix[Gaming_id]))
    
    # Sorting the cosine_similarity scores based on scores 
    cosine_scores = sorted(cosine_scores, key=lambda x:x[1], reverse = True)
    
    # Get the scores of top N most similar movies 
    cosine_scores_N = cosine_scores[0: topN+1]
    
    # Getting the movie index 
    Gaming_idx  =  [i[0] for i in cosine_scores_N]
    anime_scores =  [i[1] for i in cosine_scores_N]
    
    # Similar movies and scores
    ET_similar_show = pd.DataFrame(columns=["name", "Score"])
    ET_similar_show["name"] = Gaming1.loc[Gaming_idx, "game"]
    ET_similar_show["Score"] = anime_scores
    ET_similar_show.reset_index(inplace = True)  
    # anime_similar_show.drop(["index"], axis=1, inplace=True)
    print (ET_similar_show)
    # return (anime_similar_show)

    
# Enter your anime and number of anime's to be recommended 
get_recommendations("Super Mario Galaxy 2", topN = 5)
Gaming_index["Super Mario Galaxy 2"]

##### Best Recommended and rated game to "Super Mario Galaxy 2" are "The Legend of Zelda: Breath of the Wild" and  "Grand Theft Auto IV"



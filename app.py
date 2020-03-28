from flask import Flask,render_template,url_for,request
import pandas as pd 
import pickle
import string
#import re
import pandas as pd
import numpy as np
#from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import word_tokenize
def token(text):
    tokenized_word=word_tokenize(text)
    return tokenized_word
movies = pd.read_csv('moviesprocessed.csv')
movies.keywords = movies.keywords.astype(str).apply(token)


movies.primaryTitle  = movies.primaryTitle.astype(str).apply(lambda x : x.replace("'", ''))
titlelist = movies.primaryTitle.values.tolist()

app = Flask(__name__)
 
@app.route('/')
def home():
        
    return render_template('home.html',prediction = titlelist)



@app.route('/predict',methods=['POST'])
def predict():
    
    
    if request.method == 'POST':

        message = request.form.get('message')
                
      
        ###### helper functions. Use them when needed #######
        
        def get_poster_from_index(index):
            return movies[movies.primaryTitle == index]["poster"].values[0]
        def get_url_from_index(index):
            return movies[movies.primaryTitle == index]["URL"].values[0]

        

        def get_jaccard_sim(str1, str2):
            a = set(str1)
            b = set(str2)
            c = a.intersection(b)
            return(float(len(c)) / (len(a) + len(b) - len(c)))

        def jaccard_recommender(movie_title):
            number_of_hits = 6
            movie = movies[movies.primaryTitle==movie_title]
            keyword_string = movie.keywords.iloc[0]

            jaccards = []
            for movie in movies['keywords']:
                jaccards.append(get_jaccard_sim(keyword_string, movie))
            jaccards = pd.Series(jaccards)
            jaccards_index = jaccards.nlargest(number_of_hits+1).index
            matches = movies.loc[jaccards_index]
            movie0 = []
            for match,score in zip(matches['primaryTitle'][1:],jaccards[jaccards_index][1:]) :
                movie0.append(match)
            return movie0
        movie0 = jaccard_recommender(message)




        ## Step 7: Get a list of similar movies in descending order of similarity score
        #sorted_similar_movies = sorted(similar_movies,key=lambda x:x[1],reverse=True)


        movie1 = []
        movie2 = []
        
        for element in movie0:
            
            movie1.append(get_url_from_index(element))
            movie2.append(get_poster_from_index(element))
            



        
        

        


    return render_template('result.html',movie0=movie0,movie1=movie1,movie2=movie2)


if __name__ == '__main__':
    app.run(debug=True)
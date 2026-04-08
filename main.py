import pandas as pd
from src.preprocess import load_data
from src.sentiment_model import simple_sentiment_predict
from src.music_recommender import recommend_music

data = load_data("./dataset/sentiment140.csv")

print("Dataset Loaded Successfully")

print("\n Sentiment Driven Music Intelligence System")

user_input = input("How are you feeling today? ")

sentiment = simple_sentiment_predict(user_input)

result = recommend_music(sentiment)

print("\nDetected Sentiment:", sentiment)

print("\nRecommended Genre:")
for g in result["genre"]:
    print("-", g)

print("\nRecommended Songs:")
for s in result["songs"]:
    print("-", s)
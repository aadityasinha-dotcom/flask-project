from flask import Flask, jsonify, render_template, request
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk

nltk.download('vader_lexicon')
app = Flask(__name__)

# @app.route('/', methods=["GET", "POST"])
def main():
    # if request.method == "POST":
        inp = "I am good"
        sid = SentimentIntensityAnalyzer()
        score = sid.polarity_scores(inp)
        return score

main()
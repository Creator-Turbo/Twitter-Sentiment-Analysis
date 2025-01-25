from flask import Flask, render_template, request
import pickle
import os
import pandas as pd 

app = Flask(__name__)

# Load the pre-trained model and vectorizer
model_path = os.path.join('model', 'best_model.pkl')
vectorizer_path = os.path.join('model', 'preprocessor.pkl')

with open(model_path, 'rb') as model_file:
    model = pickle.load(model_file)

with open(vectorizer_path, 'rb') as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        tweet = request.form['tweet']
        
        # # Transform the tweet using the TF-IDF vectorizer and ensure it's a 2D array
        # tweet_vector = vectorizer.transform([[tweet]]).toarray()  # Reshape tweet to 2D
        # Convert the tweet to a DataFrame to match the expected input format
        tweet_df = pd.DataFrame({'Tweet content': [tweet]})
        
        # Predict sentiment using the trained model
        prediction = model.predict(tweet_df)
        if prediction == 3:
           sentiment = 'Positive'
        elif prediction == 2:
            sentiment = 'Neutral'
        elif prediction == 1:
            sentiment = 'Negative'
        else:
           sentiment = 'Irrelevant'
        return render_template('result.html', tweet=tweet, sentiment=sentiment)

if __name__ == '__main__':
    app.run(debug=True)

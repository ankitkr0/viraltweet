from flask import Flask, request, jsonify, render_template
from groq import Groq
import os
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file

app = Flask(__name__)
groq_client = Groq(api_key=os.getenv('GROQ_API_KEY'))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    tweet_text = data['tweetText']
    followers_count = data['followersCount']

    completion = groq_client.chat.completions.create(
        model="llama3-70b-8192",
        messages=[
            {"role": "system", "content": "I will give you a tweet URL and the number of followers I have, predict whether this tweet is going to be a hit or not, be funny, be sarcastic, be witty, Roast the tweet and person tweeting it a bit. Keep the response <50 words. End your statement with how many likes the tweet will get. Never use quotes, just give the prediction."},
            {"role": "user", "content": tweet_text},
            {"role": "assistant", "content": followers_count}
        ],
        temperature=1,
        max_tokens=1024,
        top_p=1,
        stream=True,
        stop=None,
    )

    response_text = ""
    for chunk in completion:
        response_text += chunk.choices[0].delta.content or ""

    return jsonify({"response": response_text})

if __name__ == '__main__':
    app.run(debug=True)
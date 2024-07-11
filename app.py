# importing required libraries

from feature import FeatureExtraction
from flask import Flask, request, render_template
import numpy as np

import warnings
import pickle
warnings.filterwarnings('ignore')

import pickle

# Load the saved model
with open('pickle/model.pkl', 'rb') as file:
    gbc = pickle.load(file)
file.close()


from tensorflow.keras.preprocessing.sequence import pad_sequences
import requests
from bs4 import BeautifulSoup
from tensorflow.keras.models import load_model
import nltk
import re
from nltk.corpus import stopwords
from tensorflow.keras.preprocessing.text import one_hot

def check(url):
    vocab_size = 10000
    sentence_len = 200

    models = load_model("bi_lstm_gru.h5")
    # The function take model and message as parameter


    def classify_message(model, message):
        try:
            # We will treat message as a paragraphs containing multiple sentences(lines)
            # we will extract individual lines
            for sentences in message:
                sentences = nltk.sent_tokenize(message)

                # Iterate over individual sentences
                for sentence in sentences:
                    # replace all special characters
                    words = re.sub("[^a-zA-Z]", " ", sentence)

                    # perform word tokenization of all non-english-stopwords
                    if words not in set(stopwords.words('english')):
                        word = nltk.word_tokenize(words)
                        word = " ".join(word)

            # perform one_hot on tokenized word
            oneHot = [one_hot(word, n=vocab_size)]

            # create an embedded documnet using pad_sequences
            # this can be fed to our model
            text = pad_sequences(oneHot, maxlen=sentence_len, padding="pre")

            # predict the text using model
            predict = models.predict(text)

            # if predict value is greater than 0.5 its a spam
            # if predict > 0.5:
            #     print("It is a spam")
            # # else the message is not a spam
            # else:
            #     print("It is not a spam")
            return predict
        except:
            return 0.2
            


    def scrape_website_text(url):
        # Send a GET request to the URL
        response = requests.get(url)

        # Check if the request was successful (status code 200)
        if response.status_code == 200:
            # Parse the HTML content of the page
            soup = BeautifulSoup(response.content, 'html.parser')

            # Find all text elements in the HTML
            text_elements = soup.get_text()

            return text_elements
        else:
            print(
                f"Failed to retrieve content from {url}. Status code: {response.status_code}")
            return None

    try:
        # Example usage
        url = url  # Replace with the URL of the website you want to scrape
        website_text = scrape_website_text(url)
        if website_text:
            print(website_text)
        else:
            print("Failed to scrape website text.")

        pa = website_text.split(". ")
        spam = 0
        notspam = 0
        total = 0
        for k in pa:
            print(k)
            try:
                v = classify_message(models, k)[0][0]
            except:
                v=0.2
                pass 
            if v > 0.5:

                spam += 1
            # else the message is not a spam
            else:
                notspam += 1

            total += 1
        to = notspam/total
        val = to*100
        return "Content wise Website"+str(val)+"% safe to use"
    except:
        return "unable to crab data"

app = Flask(__name__)


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":

        url = request.form["url"]
        obj = FeatureExtraction(url)
        x = np.array(obj.getFeaturesList()).reshape(1, 30)

        y_pred = gbc.predict(x)[0]
        # 1 is safe
        # -1 is unsafe
        y_pro_phishing = gbc.predict_proba(x)[0, 0]
        y_pro_non_phishing = gbc.predict_proba(x)[0, 1]
        # if(y_pred ==1 ):
        pred = "It is {0:.2f} % safe to go ".format(y_pro_phishing*100)
        data=check(url)
        print(data)
        return render_template('index.html', xx=round(y_pro_non_phishing, 2), url=url,data=data)
    return render_template("index.html", xx=-1)


if __name__ == "__main__":
    app.run(debug=True)

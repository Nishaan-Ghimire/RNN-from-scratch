import numpy as np
from flask import Flask, request, jsonify, render_template, make_response, json
from vocab_builder import build_vocab,clean_text,load_data,encode_sentences
from inference import generate_text


app = Flask(__name__)

# Load weights
weights = np.load("models/best/epoch_best.npz")
E = weights["E"]
Wx = weights["Wx"]
Wh = weights["Wh"]
b = weights["b"]
Why = weights["Why"]
by = weights["by"]
# E = np.load("models/test1/E.npy")
# Wx = np.load("models/test1/Wx.npy")
# Wh = np.load("models/test1/Wh.npy")
# Why = np.load("models/test1/Why.npy")




# Biases (if you saved them too - else just reinit zeros)
# b = np.zeros((Wx.shape[0],1))
# by = np.zeros((Why.shape[0],1))

hidden_size = Wx.shape[0]
h_prev = np.zeros((hidden_size, 1))

# Load jokes to rebuild vocab for load a saved pickle
mylines = load_data("data/jokes.txt")
tokens = []
for s in mylines:
    tokens.extend(clean_text(s))

word_to_idx, idx_to_word = build_vocab(tokens)

print(generate_text("म",15,E, Wx,Wh, b, Why, by,word_to_idx,idx_to_word,temperature=0.8,top_k=10))


@app.route("/")
def home():
    return render_template("index.html")

@app.route("/generate", methods=["POST"])
def generate():
    data = request.get_json()
    seed_word = str(data.get("seed", "म")).strip()
    num_words = int(data.get("num_words",10))
    temperature = float(data.get("temperature",1.0))
    top_k = int(data.get("top_k",5))
    text = "all okay"
    print(text)
    # print(seed_word)
    text = generate_text(seed_word, num_words, E, Wx, Wh, b, Why, by, word_to_idx, idx_to_word, temperature=1.0, top_k=5)
    response = make_response(json.dumps({"generated_text": text}, ensure_ascii=False))
    response.mimetype = "application/json"
    return response


if __name__ == "__main__":
    app.run(debug=True)
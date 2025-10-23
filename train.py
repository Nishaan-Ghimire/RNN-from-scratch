import numpy as np
from vocab_builder import load_data,tokenize,build_vocab,encode_sentences
from rnn_numpy import rnn_backward,rnn_forward,cross_entropy_loss
from generate import predict_next_word 
import os




def save_checkpoint(epoch, E, Wx, Wh, b, Why, by, path="models/checkpoints"):
    os.makedirs(path, exist_ok=True)
    np.savez(f"{path}/epoch_{epoch}.npz",
             E=E, Wx=Wx, Wh=Wh, b=b, Why=Why, by=by)
    print(f"✅ Saved checkpoint: {path}/epoch_{epoch}.npz")


def load_checkpoint(path):
    weights = np.load(path)
    return (weights["E"], weights["Wx"], weights["Wh"],
            weights["b"], weights["Why"], weights["by"])

# Load data
mylines = load_data()

# Tokenize the data
tokens = tokenize(mylines)

# Vocabulary building
word_to_idx, idx_to_word = build_vocab(tokens)

# Encode to numbers
encoded = encode_sentences(mylines,word_to_idx)


# Start initializing the parameters for RNN training


vocab_size = len(word_to_idx)
embedding_dim = 50
hidden_size = 50
learning_rate = 0.01

# print("Vocab Size : ",vocab_size)

# RNN weights
Wx = np.random.randn(hidden_size,embedding_dim) * 0.01 # input -> hidden
Wh = np.random.randn(hidden_size,hidden_size) * 0.01 # hidden -> hidden
b = np.zeros((hidden_size,1))

# Output weights
Why = np.random.randn(vocab_size,hidden_size) * 0.01 # hidden -> vocab
by = np.zeros((vocab_size,1))                        # output bias



# Initialize embeddings randomly
E = np.random.randn(vocab_size,embedding_dim) * 0.01

# Example: convert indices to vectors
# sample_indices = [256,285, 166]
h_prev = np.zeros((hidden_size,1))



# Step: prepare gradients placeholders for BPTT

dWx = np.zeros_like(Wx)
dWh = np.zeros_like(Wh)
db = np.zeros_like(b)
dWhy = np.zeros_like(Why)
dby = np.zeros_like(by)
dE = np.zeros_like(E)


epochs = 1000
save_every = 50
best_loss = float("inf")


for epoch in range(epochs):
    total_loss = 0
    for joke in encoded:
        if len(joke) < 2:
            continue
        X = joke[:-1] # input words
        Y = joke[1:]  # target next words
        X_embedded = E[X]

        hs, ys = rnn_forward(X_embedded,h_prev,Wx,Wh,b,Why,by)

        # compute loss
        for t in range(len(Y)):
            y_logits = ys[t]
            exp_scores = np.exp(y_logits)
            p_t = exp_scores / np.sum(exp_scores)
            total_loss += cross_entropy_loss(p_t, Y[t])

        # Backpropagation 
        Wx, Wh, b, Why, by, E = rnn_backward(X_embedded, Y, hs, ys, Wx, Wh, b, Why, by, E, learning_rate)

    avg_loss = total_loss / len(encoded)
    print(f"Epoch {epoch+1}, Average Loss: {avg_loss:.4f}")

    # Save best model if loss improves 
    if avg_loss < best_loss:
        best_loss = avg_loss
        save_checkpoint("best",E, Wx,Wh,b,Why,by, path="models/best")
        
        # run quick inference on best model
        print("\n🧩 Sample generation (best model):")
        print(predict_next_word("म", E, Wx, Wh, b, Why, by, word_to_idx, idx_to_word, hidden_size))
        print("--------------------------------------------------")

    # Regular periodic checkpoints
    if epoch % save_every == 0:
        save_checkpoint(epoch,E,Wx,Wh,b,Why,by)
        # inference at checkpoints
        print(f"\n📜 Epoch {epoch} checkpoint sample:")
        print(predict_next_word("मेरो", E, Wx, Wh, b, Why, by, word_to_idx, idx_to_word, hidden_size))
        print("--------------------------------------------------")


# np.save("models/test1/E.npy", E)
# np.save("models/test1/Wx.npy", Wx)
# np.save("models/test1/Wh.npy", Wh)
# np.save("models/test1/Why.npy", Why)
# print("\n--- Testing Model ---")
# test_sentence = predict_next_word("राम", E, Wx, Wh, b, Why, by, word_to_idx, idx_to_word, hidden_size)

# print(test_sentence)

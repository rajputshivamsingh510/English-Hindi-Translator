import os
# Skip CUDA probing entirely (saves a few seconds on every cold start,
# and avoids noisy "could not find cuda drivers" log spam).
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
import tensorflow as tf

from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import Input
from tensorflow.keras.preprocessing.sequence import pad_sequences

app = Flask(__name__)

# ==========================
# Configuration
# ==========================

MAX_ENG_LEN = 18
MAX_HIN_LEN = 18
LSTM_UNITS = 128

MODEL_PATH = "best_translations_model.keras"
ENG_TOKENIZER_PATH = "eng_tokenizer.pkl"
HIN_TOKENIZER_PATH = "hin_tokenizer.pkl"

# ==========================
# Load Model & Tokenizers (once, at startup)
# ==========================

trained_model = load_model(MODEL_PATH)

with open(ENG_TOKENIZER_PATH, "rb") as f:
    eng_tokenizer = pickle.load(f)

with open(HIN_TOKENIZER_PATH, "rb") as f:
    hin_tokenizer = pickle.load(f)

# ==========================
# Build Encoder
# ==========================

encoder_inputs = trained_model.input[0]

_, state_h, state_c = trained_model.get_layer("Encoder_LSTM").output

encoder_model = Model(encoder_inputs, [state_h, state_c])

# ==========================
# Build Decoder
# ==========================

decoder_embedding_layer = trained_model.get_layer("Decoder_Embedding")
decoder_lstm = trained_model.get_layer("Decoder_LSTM")
decoder_dense = trained_model.get_layer("Output_Layer")

decoder_state_input_h = Input(shape=(LSTM_UNITS,))
decoder_state_input_c = Input(shape=(LSTM_UNITS,))
decoder_single_input = Input(shape=(1,))

decoder_embedding = decoder_embedding_layer(decoder_single_input)

decoder_outputs, state_h2, state_c2 = decoder_lstm(
    decoder_embedding,
    initial_state=[decoder_state_input_h, decoder_state_input_c],
)

decoder_outputs = decoder_dense(decoder_outputs)

decoder_model = Model(
    [decoder_single_input, decoder_state_input_h, decoder_state_input_c],
    [decoder_outputs, state_h2, state_c2],
)

# ==========================
# Warm up (forces TF to trace the graphs ONCE at startup, on the
# server's own time, instead of on the first user's request)
# ==========================

_dummy_seq = np.zeros((1, MAX_ENG_LEN), dtype="int32")
_dummy_states = encoder_model(_dummy_seq, training=False)
_dummy_target = np.zeros((1, 1), dtype="int32")
decoder_model([_dummy_target] + list(_dummy_states), training=False)

# ==========================
# Translation Function
# ==========================

def translate(sentence):
    sentence = sentence.lower().strip()

    seq = eng_tokenizer.texts_to_sequences([sentence])
    seq = pad_sequences(seq, maxlen=MAX_ENG_LEN, padding="post", truncating="post")

    # Direct call instead of .predict() -- avoids Keras's per-call
    # data-pipeline/retracing overhead, which is what was making each
    # translation take minutes instead of milliseconds.
    states_value = list(encoder_model(seq, training=False))

    start_token = hin_tokenizer.word_index["<start>"]
    target_seq = np.array([[start_token]])

    translated_words = []

    for _ in range(MAX_HIN_LEN):
        output_tokens, h, c = decoder_model(
            [target_seq] + states_value, training=False
        )

        sampled_index = int(np.argmax(output_tokens[0, -1, :].numpy()))
        sampled_word = hin_tokenizer.index_word.get(sampled_index, "")

        if sampled_word == "<end>" or sampled_index == 0:
            break

        translated_words.append(sampled_word)

        target_seq = np.array([[sampled_index]])
        states_value = [h, c]

    return " ".join(translated_words)

# ==========================
# Routes
# ==========================

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/translate", methods=["POST"])
def translate_api():
    data = request.get_json()
    english = data["text"]
    hindi = translate(english)
    return jsonify({"translation": hindi})


@app.route("/healthz")
def healthz():
    return "ok"

# ==========================

if __name__ == "__main__":
    app.run(debug=True)

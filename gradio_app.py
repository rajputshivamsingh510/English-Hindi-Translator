import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import pickle
import numpy as np
import gradio as gr

from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import Input
from tensorflow.keras.preprocessing.sequence import pad_sequences

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
# Warm up (trace the graphs once at startup, not on first user request)
# ==========================

_dummy_seq = np.zeros((1, MAX_ENG_LEN), dtype="int32")
_dummy_states = [s.numpy() for s in encoder_model(_dummy_seq, training=False)]
_dummy_target = np.zeros((1, 1), dtype="int32")
decoder_model([_dummy_target] + _dummy_states, training=False)

# ==========================
# Translation Function
# ==========================

def translate(sentence):
    if not sentence or not sentence.strip():
        return ""

    sentence = sentence.lower().strip()

    seq = eng_tokenizer.texts_to_sequences([sentence])
    seq = pad_sequences(seq, maxlen=MAX_ENG_LEN, padding="post", truncating="post")

    states_value = [s.numpy() for s in encoder_model(seq, training=False)]

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
        states_value = [h.numpy(), c.numpy()]

    return " ".join(translated_words)

# ==========================
# Gradio UI
# ==========================

demo = gr.Interface(
    fn=translate,
    inputs=gr.Textbox(label="English", placeholder="Type an English sentence..."),
    outputs=gr.Textbox(label="Hindi"),
    title="English → Hindi Translator",
    description="A seq2seq LSTM neural machine translation model.",
)

if __name__ == "__main__":
    demo.launch()

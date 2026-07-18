"""
Test script for the English -> Hindi seq2seq LSTM translator.

Requires, in the same folder:
  - best_translation_model.keras   (your trained model)
  - eng_tokenizer.pkl
  - hin_tokenizer.pkl

Run:
  python test_translator.py
"""

import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input
from tensorflow.keras.preprocessing.sequence import pad_sequences

# --- Must match the values used during training ---
MAX_ENG_LEN = 18
MAX_HIN_LEN = 18
LSTM_UNITS = 128

MODEL_PATH = "best_translations_model.keras"
ENG_TOKENIZER_PATH = "eng_tokenizer.pkl"
HIN_TOKENIZER_PATH = "hin_tokenizer.pkl"


def load_assets():
    trained_model = load_model(MODEL_PATH)

    with open(ENG_TOKENIZER_PATH, "rb") as f:
        eng_tokenizer = pickle.load(f)
    with open(HIN_TOKENIZER_PATH, "rb") as f:
        hin_tokenizer = pickle.load(f)

    return trained_model, eng_tokenizer, hin_tokenizer


def build_inference_models(trained_model):
    """
    Rebuilds encoder + step-by-step decoder models using the SAME trained
    layers (so weights are reused, no retraining needed).
    """
    # --- Encoder inference model ---
    # Pull tensors directly off the already-built graph (don't reconstruct
    # from a bare layer's `.input`/`.output` — unreliable across Keras versions).
    encoder_inputs = trained_model.input[0]  # the "Encoder_Input" tensor
    _, state_h, state_c = trained_model.get_layer("Encoder_LSTM").output
    encoder_model = Model(encoder_inputs, [state_h, state_c])

    # --- Decoder inference model (runs one token at a time) ---
    decoder_embedding_layer = trained_model.get_layer("Decoder_Embedding")
    decoder_lstm = trained_model.get_layer("Decoder_LSTM")
    decoder_dense = trained_model.get_layer("Output_Layer")

    decoder_state_input_h = Input(shape=(LSTM_UNITS,), name="decoder_state_input_h")
    decoder_state_input_c = Input(shape=(LSTM_UNITS,), name="decoder_state_input_c")
    decoder_single_input = Input(shape=(1,), name="decoder_single_input")

    dec_emb2 = decoder_embedding_layer(decoder_single_input)
    decoder_outputs2, state_h2, state_c2 = decoder_lstm(
        dec_emb2, initial_state=[decoder_state_input_h, decoder_state_input_c]
    )
    decoder_outputs2 = decoder_dense(decoder_outputs2)

    decoder_model = Model(
        [decoder_single_input, decoder_state_input_h, decoder_state_input_c],
        [decoder_outputs2, state_h2, state_c2],
    )

    return encoder_model, decoder_model


def translate(sentence, encoder_model, decoder_model, eng_tokenizer, hin_tokenizer):
    sentence = sentence.lower().strip()
    seq = eng_tokenizer.texts_to_sequences([sentence])
    seq = pad_sequences(seq, maxlen=MAX_ENG_LEN, padding="post", truncating="post")

    states_value = encoder_model.predict(seq, verbose=0)

    start_token = hin_tokenizer.word_index.get("<start>")
    end_token = hin_tokenizer.word_index.get("<end>")

    target_seq = np.array([[start_token]])

    decoded_words = []
    for _ in range(MAX_HIN_LEN):
        output_tokens, h, c = decoder_model.predict([target_seq] + states_value, verbose=0)

        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_word = hin_tokenizer.index_word.get(sampled_token_index, "<unk>")

        if sampled_word == "<end>" or sampled_token_index == 0:
            break

        decoded_words.append(sampled_word)

        target_seq = np.array([[sampled_token_index]])
        states_value = [h, c]

    return " ".join(decoded_words)


def main():
    trained_model, eng_tokenizer, hin_tokenizer = load_assets()
    encoder_model, decoder_model = build_inference_models(trained_model)

    test_sentences = [
        "how are you",
        "i am fine",
        "good morning",
        "what is your name",
        "the weather is nice today",
    ]

    print("Testing translations:\n")
    for s in test_sentences:
        translation = translate(s, encoder_model, decoder_model, eng_tokenizer, hin_tokenizer)
        print(f"EN: {s}")
        print(f"HI: {translation}\n")

    # Interactive mode
    print("Type a sentence to translate (or 'quit' to exit):")
    while True:
        user_input = input("> ").strip()
        if user_input.lower() in ("quit", "exit"):
            break
        print(translate(user_input, encoder_model, decoder_model, eng_tokenizer, hin_tokenizer))


if __name__ == "__main__":
    main()
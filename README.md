# English → Hindi Translator

A sequence-to-sequence LSTM neural machine translation model that translates
English sentences into Hindi. Trained in `translator_fast (1).ipynb`.

**🌐 Live demo:** https://huggingface.co/spaces/beast0X1/english_to_hindi_translator

---

## Run it locally

```bash
pip install -r requirements.txt
python app.py
```

Then open **http://127.0.0.1:5000** in your browser.

Make sure these files are present alongside `app.py` (already included in
this repo):
- `best_translations_model.keras`
- `eng_tokenizer.pkl`
- `hin_tokenizer.pkl`

## Deploying it yourself

Run it behind gunicorn instead of the Flask dev server:
```bash
gunicorn app:app --timeout 120
```

## Model details

- Architecture: Encoder-decoder LSTM (embedding dim 64, 128 LSTM units, ~10k
  vocabulary), trained with teacher forcing.
- See `translator_fast (1).ipynb` for the full training pipeline, or
  `test_translator.py` for a minimal CLI test script using the same
  inference logic as `app.py`.

## A note on accuracy

This model was trained on a limited subset of the dataset with a
comparatively small architecture, due to compute/memory constraints on the
hardware used for training. As a result, translations won't be as accurate
or fluent as production-grade translators — expect it to work reasonably
well on short, simple sentences, and to struggle more with longer or more
complex ones. Training on more data with a larger model would improve
translation quality, but requires significantly more compute than was
available here.

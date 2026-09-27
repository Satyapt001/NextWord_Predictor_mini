from pathlib import Path
import pickle

import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences


class NextWordPredictor:
    """Hybrid N-gram + LSTM next-word predictor."""

    def __init__(self, model_dir):
        self.model_dir = Path(model_dir)

        self.model = load_model(
            self.model_dir / "next_word_predictor.keras"
        )

        with open(
            self.model_dir / "tokenizer.pkl",
            "rb"
        ) as file:
            self.tokenizer = pickle.load(file)

        with open(
            self.model_dir / "ngrams.pkl",
            "rb"
        ) as file:
            self.ngram_counts = pickle.load(file)

        self.index_word = {
            index: word
            for word, index
            in self.tokenizer.word_index.items()
        }

        self.max_sequence_len = (
            self.model.input_shape[1] + 1
        )

    # --------------------------------------------------------
    # Text cleaning
    # --------------------------------------------------------

    @staticmethod
    def clean_text(text):
        return " ".join(
            text.lower().strip().split()
        )

    # --------------------------------------------------------
    # N-gram prediction
    # --------------------------------------------------------

    def predict_ngram(self, text, top_k=5):

        words = self.clean_text(text).split()

        if len(words) < 2:
            return []

        for n in [5, 4, 3, 2]:

            if len(words) < n:
                continue

            context = tuple(words[-n:])

            candidates = (
                self.ngram_counts
                .get(n, {})
                .get(context)
            )

            if not candidates:
                continue

            ranked = sorted(
                candidates.items(),
                key=lambda item: item[1],
                reverse=True
            )

            return [
                word
                for word, _ in ranked[:top_k]
            ]

        return []

    # --------------------------------------------------------
    # LSTM prediction
    # --------------------------------------------------------

    def predict_lstm(self, text, top_k=5):

        text = self.clean_text(text)

        if not text:
            return []

        sequence = (
            self.tokenizer
            .texts_to_sequences([text])[0]
        )

        oov_id = self.tokenizer.word_index.get(
            "<OOV>"
        )

        if oov_id:
            sequence = [
                token
                for token in sequence
                if token != oov_id
            ]

        if not sequence:
            return []

        sequence = pad_sequences(
            [sequence],
            maxlen=self.max_sequence_len - 1,
            padding="pre"
        )

        probabilities = self.model.predict(
            sequence,
            verbose=0
        )[0]

        candidate_ids = np.argsort(
            probabilities
        )[::-1]

        results = []

        for word_id in candidate_ids:

            word = self.index_word.get(
                word_id
            )

            if not word:
                continue

            if word == "<OOV>":
                continue

            results.append(word)

            if len(results) >= top_k:
                break

        return results

    # --------------------------------------------------------
    # Hybrid prediction
    # --------------------------------------------------------

    def predict(self, text, top_k=5):

        if not text.strip():
            return []

        ngram_results = self.predict_ngram(
            text,
            top_k
        )

        if ngram_results:
            return ngram_results

        return self.predict_lstm(
            text,
            top_k
        )
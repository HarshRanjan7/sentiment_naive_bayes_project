"""
Multinomial Naive Bayes implemented from scratch.

No scikit-learn is used.

The model estimates:
P(word | class)

with Laplace smoothing and combines the word probabilities
to estimate the most likely sentiment.
"""

import math
from collections import Counter
from preprocess import tokenize


class NaiveBayes:
    def __init__(self):
        self.class_counts = Counter()
        self.word_counts = {
            0: Counter(),
            1: Counter()
        }
        self.total_words = {
            0: 0,
            1: 0
        }
        self.vocabulary = set()
        self.total_documents = 0

    def fit(self, data):
        self.total_documents = len(data)

        for sentence, label in data:
            self.class_counts[label] += 1

            words = tokenize(sentence)

            for word in words:
                self.word_counts[label][word] += 1
                self.total_words[label] += 1
                self.vocabulary.add(word)

    def word_probability(self, word, label):
        # Laplace smoothing prevents zero probability.
        count = self.word_counts[label][word]
        vocabulary_size = len(self.vocabulary)

        return (count + 1) / (
            self.total_words[label] + vocabulary_size
        )

    def class_probability(self, label):
        return self.class_counts[label] / self.total_documents

    def predict_proba(self, sentence):
        words = tokenize(sentence)

        scores = {}

        for label in [0, 1]:
            # Use logarithms to avoid multiplying many tiny probabilities.
            score = math.log(self.class_probability(label))

            for word in words:
                score += math.log(
                    self.word_probability(word, label)
                )

            scores[label] = score

        # Convert log-scores into normalized probabilities.
        max_score = max(scores.values())

        exp_scores = {
            label: math.exp(score - max_score)
            for label, score in scores.items()
        }

        total = sum(exp_scores.values())

        return {
            label: exp_scores[label] / total
            for label in [0, 1]
        }

    def predict(self, sentence):
        probabilities = self.predict_proba(sentence)

        return max(
            probabilities,
            key=probabilities.get
        )

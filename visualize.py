"""
Creates a confusion-matrix-style result visualization.

The model is trained on the small dataset and tested on six
new labeled sentences.
"""

import matplotlib.pyplot as plt
from dataset import data
from naive_bayes import NaiveBayes

model = NaiveBayes()
model.fit(data)

test_data = [
    ("I love this film", 1),
    ("This product is fantastic", 1),
    ("The experience was horrible", 0),
    ("I hate this product", 0),
    ("The movie was pleasant", 1),
    ("This was boring", 0),
]

# Confusion matrix:
# rows = actual, columns = predicted
matrix = [
    [0, 0],  # actual negative
    [0, 0],  # actual positive
]

for sentence, actual in test_data:
    predicted = model.predict(sentence)
    matrix[actual][predicted] += 1

plt.figure(figsize=(7, 6))
plt.imshow(matrix, cmap="Blues")

plt.title("Sentiment Analysis - Naive Bayes")
plt.xlabel("Predicted Class")
plt.ylabel("Actual Class")

plt.xticks([0, 1], ["Negative", "Positive"])
plt.yticks([0, 1], ["Negative", "Positive"])

for row in range(2):
    for col in range(2):
        plt.text(
            col, row,
            matrix[row][col],
            ha="center",
            va="center",
            fontsize=20
        )

plt.colorbar(label="Number of sentences")
plt.tight_layout()

plt.savefig(
    "sentiment_naive_bayes_result.png",
    dpi=180
)

plt.show()

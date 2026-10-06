"""
Main program for the Sentiment Analysis project.
"""

from dataset import data
from naive_bayes import NaiveBayes

model = NaiveBayes()
model.fit(data)

print("=" * 60)
print("       SENTIMENT ANALYSIS USING NAIVE BAYES")
print("=" * 60)

print("\nTraining information:")
print("Total sentences:", len(data))
print("Positive sentences:", model.class_counts[1])
print("Negative sentences:", model.class_counts[0])
print("Vocabulary size:", len(model.vocabulary))

test_sentences = [
    "I love this film",
    "This product is fantastic",
    "The experience was horrible",
    "I hate this product",
    "The movie was pleasant",
    "This was boring"
]

print("\nPredictions:")
print("-" * 60)

correct = 0

for sentence, actual in [
    ("I love this film", 1),
    ("This product is fantastic", 1),
    ("The experience was horrible", 0),
    ("I hate this product", 0),
    ("The movie was pleasant", 1),
    ("This was boring", 0)
]:
    probabilities = model.predict_proba(sentence)
    prediction = model.predict(sentence)

    predicted_name = "POSITIVE" if prediction == 1 else "NEGATIVE"

    print(
        f"{sentence:32s} -> "
        f"{predicted_name:8s} "
        f"(positive={probabilities[1]:.2f}, "
        f"negative={probabilities[0]:.2f})"
    )

    if prediction == actual:
        correct += 1

accuracy = correct / 6 * 100

print("\nTest accuracy:", f"{accuracy:.1f}%")

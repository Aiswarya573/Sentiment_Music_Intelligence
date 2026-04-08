import numpy as np
import math
from collections import defaultdict


# ─────────────────────────────────────────────
#  Expanded Lexicon (rule-based fallback)
# ─────────────────────────────────────────────

POSITIVE_WORDS = [
    "happy", "love", "great", "good", "awesome", "excellent", "fantastic",
    "wonderful", "amazing", "joyful", "excited", "cheerful", "thrilled",
    "delighted", "brilliant", "superb", "beautiful", "nice", "glad",
    "pleased", "ecstatic", "elated", "grateful", "blessed", "peaceful",
    "content", "hopeful", "radiant", "fun", "enjoy", "enjoyed", "smile",
    "laugh", "energetic", "motivated", "inspired", "proud", "confident",
    "fabulous", "magnificent", "incredible", "outstanding", "perfect",
    "positive", "optimistic", "lively", "playful", "refreshed", "relaxed"
]

NEGATIVE_WORDS = [
    "sad", "bad", "angry", "hate", "terrible", "horrible", "awful",
    "depressed", "miserable", "unhappy", "upset", "frustrated", "annoyed",
    "disgusted", "furious", "devastated", "heartbroken", "lonely",
    "anxious", "worried", "stressed", "exhausted", "tired", "bored",
    "hopeless", "helpless", "pathetic", "useless", "worthless", "lost",
    "confused", "scared", "fearful", "nervous", "gloomy", "melancholy",
    "bitter", "resentful", "jealous", "envious", "guilty", "ashamed",
    "embarrassed", "rejected", "abandoned", "betrayed", "hurt", "pain",
    "crying", "cry", "tears", "dark", "negative", "pessimistic"
]

INTENSIFIERS = ["very", "really", "so", "extremely", "absolutely", "totally", "super"]
NEGATIONS    = ["not", "never", "no", "don't", "doesn't", "didn't", "isn't", "wasn't", "can't", "won't"]


# ─────────────────────────────────────────────
#  Naïve Bayes Classifier
# ─────────────────────────────────────────────

class NaiveBayesSentiment:
    """
    Multinomial Naïve Bayes trained on a small seed corpus.
    Falls back gracefully to the lexicon when confidence is low.
    """

    def __init__(self):
        self.class_log_prior   = {}
        self.feature_log_prob  = {}
        self.classes           = ["positive", "neutral", "negative"]
        self.vocab             = set()
        self._train()

    # ── seed training data ──────────────────────
    def _get_seed_data(self):
        data = []

        pos = [
            "I am so happy today", "This is great and awesome",
            "I love this wonderful day", "Feeling fantastic and excited",
            "Amazing experience, totally brilliant", "I am joyful and grateful",
            "Had an excellent and fun time", "Life is beautiful and peaceful",
            "Feeling blessed and content", "I am thrilled and delighted",
            "Everything is perfect and wonderful", "I feel inspired and motivated",
            "Great day, feeling cheerful", "Absolutely incredible moment",
            "I enjoyed every bit of it", "Smiling and feeling refreshed",
        ]
        neg = [
            "I am so sad and upset", "This is terrible and horrible",
            "I hate this awful situation", "Feeling depressed and miserable",
            "Exhausted and completely stressed out", "I am frustrated and angry",
            "This is disgusting and bad", "Feeling hopeless and lost",
            "I am heartbroken and lonely", "Everything went wrong today",
            "I feel worthless and rejected", "So anxious and nervous",
            "Crying and feeling devastated", "I am bitter and resentful",
            "Dark and gloomy thoughts", "Feeling scared and helpless",
        ]
        neu = [
            "Today was an ordinary day", "Nothing special happened",
            "I went to the store and came back", "The weather is cloudy",
            "I read a book this afternoon", "Had lunch and took a nap",
            "Watched some TV shows", "Just another regular day",
            "Things are the same as usual", "Not much going on today",
            "Did some work and rested", "The meeting was average",
        ]

        for s in pos: data.append((s, "positive"))
        for s in neg: data.append((s, "negative"))
        for s in neu: data.append((s, "neutral"))
        return data

    # ── fit ─────────────────────────────────────
    def _train(self):
        data   = self._get_seed_data()
        counts = defaultdict(lambda: defaultdict(int))
        class_counts = defaultdict(int)

        for text, label in data:
            class_counts[label] += 1
            for word in text.lower().split():
                counts[label][word] += 1
                self.vocab.add(word)

        total = len(data)
        vocab_size = len(self.vocab)

        for cls in self.classes:
            self.class_log_prior[cls] = math.log(class_counts[cls] / total)
            total_words = sum(counts[cls].values())
            self.feature_log_prob[cls] = {}
            for word in self.vocab:
                # Laplace smoothing
                self.feature_log_prob[cls][word] = math.log(
                    (counts[cls][word] + 1) / (total_words + vocab_size)
                )
            self.feature_log_prob[cls]["<UNK>"] = math.log(1 / (total_words + vocab_size))

    # ── predict ─────────────────────────────────
    def predict(self, text):
        words  = text.lower().split()
        scores = {}
        for cls in self.classes:
            score = self.class_log_prior[cls]
            for word in words:
                score += self.feature_log_prob[cls].get(
                    word, self.feature_log_prob[cls]["<UNK>"]
                )
            scores[cls] = score

        # Convert log-probs → probabilities via softmax
        max_s  = max(scores.values())
        exp_s  = {c: math.exp(scores[c] - max_s) for c in self.classes}
        total  = sum(exp_s.values())
        probs  = {c: exp_s[c] / total for c in self.classes}

        best   = max(probs, key=probs.get)
        confidence = probs[best]
        return best, confidence, probs


# ─────────────────────────────────────────────
#  Lexicon-based scorer (handles negation)
# ─────────────────────────────────────────────

def _lexicon_score(tokens):
    pos = neg = 0
    negate = False
    intensity = 1.0

    for word in tokens:
        if word in NEGATIONS:
            negate = True
            continue
        if word in INTENSIFIERS:
            intensity = 1.5
            continue

        if word in POSITIVE_WORDS:
            if negate:
                neg += intensity
            else:
                pos += intensity
            negate = False
            intensity = 1.0

        elif word in NEGATIVE_WORDS:
            if negate:
                pos += intensity
            else:
                neg += intensity
            negate = False
            intensity = 1.0

    return pos - neg


# ─────────────────────────────────────────────
#  Public API
# ─────────────────────────────────────────────

_nb_model = NaiveBayesSentiment()


def simple_sentiment_predict(text: str) -> str:
    """Return 'positive', 'neutral', or 'negative'."""
    label, _, _ = predict_with_confidence(text)
    return label


def predict_with_confidence(text: str):
    """
    Returns (label, confidence_pct, detail_dict).
    Combines Naïve Bayes + lexicon for robustness.
    """
    tokens = text.lower().split()

    # Naïve Bayes
    nb_label, nb_conf, nb_probs = _nb_model.predict(text)

    # Lexicon
    lex_score = _lexicon_score(tokens)
    if lex_score > 0:
        lex_label = "positive"
    elif lex_score < 0:
        lex_label = "negative"
    else:
        lex_label = "neutral"

    # Ensemble: if both agree → high confidence
    if nb_label == lex_label:
        final_label = nb_label
        confidence  = min(nb_conf * 1.15, 1.0)
    else:
        # Trust NB when confident, else trust lexicon
        final_label = nb_label if nb_conf > 0.55 else lex_label
        confidence  = nb_conf if nb_conf > 0.55 else 0.50

    confidence_pct = round(confidence * 100, 1)

    detail = {
        "label":          final_label,
        "confidence_pct": confidence_pct,
        "nb_label":       nb_label,
        "nb_probs":       {k: round(v * 100, 1) for k, v in nb_probs.items()},
        "lexicon_score":  lex_score,
        "lexicon_label":  lex_label,
    }
    return final_label, confidence_pct, detail

"""
Baseline bigram next-word generator (pure Python, no external deps).
Trains a simple conditional frequency model on `data.txt` and generates text word-by-word.
"""
import random
import json
from collections import defaultdict, Counter

DATA_PATH = "data.txt"
MODEL_PATH = "bigram_model.json"


def tokenize(text):
    # simple whitespace tokenizer, keep punctuation attached
    return text.strip().split()


def train_bigram(data_path=DATA_PATH):
    counts = defaultdict(Counter)
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            words = tokenize(line)
            if not words:
                continue
            words = ["<s>"] + words + ["</s>"]
            for a, b in zip(words, words[1:]):
                counts[a][b] += 1
    # convert to probabilities
    model = {w: dict(counter) for w, counter in counts.items()}
    with open(MODEL_PATH, "w", encoding="utf-8") as out:
        json.dump(model, out, ensure_ascii=False, indent=2)
    return model


def load_model(path=MODEL_PATH):
    with open(path, "r", encoding="utf-8") as f:
        model = json.load(f)
    return {k: Counter(v) for k, v in model.items()}


def sample_next(counter):
    items = list(counter.items())
    total = sum(c for _, c in items)
    r = random.randint(1, total)
    upto = 0
    for w, c in items:
        upto += c
        if upto >= r:
            return w
    return items[-1][0]


def generate(model, prompt=None, max_words=20):
    if prompt:
        words = tokenize(prompt)
        cur = words[-1]
    else:
        cur = "<s>"
        words = []
    for _ in range(max_words):
        counter = model.get(cur) or model.get("<s>")
        if not counter:
            break
        nxt = sample_next(counter)
        if nxt == "</s>":
            break
        words.append(nxt)
        cur = nxt
    return " ".join(words)


if __name__ == "__main__":
    print("Training bigram model on data.txt...")
    model = train_bigram()
    print("Model saved to", MODEL_PATH)
    loaded = load_model()
    print("Generating samples:")
    for prompt in [None, "I love to", "I like to", "Coding and"]:
        print(f"Prompt: {prompt}")
        print(generate(loaded, prompt=prompt, max_words=10))
        print()

from transformers import AutoTokenizer
from mamkit.data.datasets import MMUSEDFallacy, InputMode
from pathlib import Path
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
import re
import nltk
from nltk.corpus import stopwords

# Download stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# BERT tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# Label mapping
label_map = {
    0: 'Appeal to Emotion',
    1: 'Appeal to Authority',
    2: 'Ad Hominem',
    3: 'False Cause',
    4: 'Slippery Slope',
    5: 'Slogans'
}

# Load AFC dataset
loader = MMUSEDFallacy(
    task_name='afc',
    input_mode=InputMode.TEXT_AUDIO,
    base_data_path=Path(__file__).parent.parent.resolve().joinpath('data'),
    with_context=False
)
split_info = next(loader.get_splits(key='mancini-et-al-2024'))
dataset = list(split_info.train) + list(split_info.val) + list(split_info.test)

# Collect tokens per fallacy class
tokens_per_class = defaultdict(list)

for text, _, label, *_ in dataset:
    tokens = tokenizer.tokenize(text.lower())
    for tok in tokens:
        if tok in stop_words: continue
        if re.fullmatch(r"[\W_]+", tok): continue
        if tok.isnumeric(): continue
        tokens_per_class[label].append(tok)

# Plot top-k tokens per label
top_k = 20
for label in range(6):
    token_counts = Counter(tokens_per_class[label])
    print(f"\n🔍 Top {top_k} tokens for label {label}: {label_map[label]}")
    for token, count in token_counts.most_common(top_k):
        print(f"{token}: {count}")

    # Plot
    top_items = token_counts.most_common(top_k)
    words, counts = zip(*top_items)
    plt.figure(figsize=(10, 4))
    plt.bar(words, counts, color='teal')
    plt.title(f"Top {top_k} Tokens - {label_map[label]}")
    plt.xticks(rotation=45)
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.show()
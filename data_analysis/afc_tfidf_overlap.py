from transformers import AutoTokenizer
from mamkit.data.datasets import MMUSEDFallacy, InputMode
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.cm as cm # <--- ADD THIS IMPORT

# Label mapping
label_map = {
    0: 'Appeal to Emotion',
    1: 'Appeal to Authority',
    2: 'Ad Hominem',
    3: 'False Cause',
    4: 'Slippery Slope',
    5: 'Slogans'
}

# Load dataset
loader = MMUSEDFallacy(
    task_name='afc',
    input_mode=InputMode.TEXT_AUDIO,
    base_data_path=Path(__file__).parent.parent.resolve().joinpath('data'),
    with_context=False
)
split_info = next(loader.get_splits(key='mancini-et-al-2024'))
dataset = list(split_info.train) + list(split_info.val) + list(split_info.test)

# Collect texts and labels
texts = [ex[0] for ex in dataset if ex[2] is not None]
labels = [label_map[ex[2]] for ex in dataset if ex[2] is not None]

df = pd.DataFrame({'text': texts, 'label': labels})

# TF-IDF Token Extraction
class_tfidf_tokens = defaultdict(list)
tfidf_results = {}

for label in df['label'].unique():
    subset = df[df['label'] == label]
    vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
    X = vectorizer.fit_transform(subset['text'])
    means = X.mean(axis=0).A1
    tokens = vectorizer.get_feature_names_out()
    tfidf_scores = sorted(zip(tokens, means), key=lambda x: x[1], reverse=True)[:20]
    tfidf_results[label] = tfidf_scores
    class_tfidf_tokens[label] = [token for token, _ in tfidf_scores]

# Plot per-class TF-IDF tokens
fig, axs = plt.subplots(2, 3, figsize=(18, 10))
axs = axs.ravel()

for idx, (label, scores) in enumerate(tfidf_results.items()):
    tokens, values = zip(*scores)

    # Generate colors for the current subplot using the viridis colormap
    num_tokens_in_subplot = len(tokens)

    subplot_colors = [cm.viridis(i / num_tokens_in_subplot) for i in range(num_tokens_in_subplot)]

    axs[idx].bar(tokens, values, color=subplot_colors) # <--- USE THE GENERATED COLORS

    axs[idx].set_title(label)
    axs[idx].tick_params(axis='x', rotation=45)
    axs[idx].set_ylabel('Avg TF-IDF Score')

plt.tight_layout()
plt.suptitle("Top TF-IDF Tokens per Fallacy Class", fontsize=16, y=1.05)
plt.show()

# Overlap Matrix
token_sets = {label: set(tokens) for label, tokens in class_tfidf_tokens.items()}
labels_list = list(token_sets.keys())
overlap_matrix = []

for l1 in labels_list:
    row = []
    for l2 in labels_list:
        row.append(len(token_sets[l1].intersection(token_sets[l2])))
    overlap_matrix.append(row)

overlap_df = pd.DataFrame(overlap_matrix, index=labels_list, columns=labels_list)

plt.figure(figsize=(8, 6))
sns.heatmap(overlap_df, annot=True, cmap='Blues', fmt='d')
plt.title("Token Overlap Across Fallacy Classes (Top 20 TF-IDF)")
plt.show()


def print_overlap_tokens(class1, class2):
    set1 = token_sets.get(class1)
    set2 = token_sets.get(class2)
    if set1 is None or set2 is None:
        print(f"Invalid class names. Choose from: {list(token_sets.keys())}")
        return
    overlap = sorted(set1.intersection(set2))
    print(f"\n🔁 Overlapping tokens between '{class1}' and '{class2}':")
    if overlap:
        print(", ".join(overlap))
    else:
        print("No overlap.")

print("\n🔍 Overlapping TF-IDF Tokens Between All Class Pairs:\n")

for i, class1 in enumerate(labels_list):
    for j, class2 in enumerate(labels_list):
        if i >= j:
            continue  # skip duplicates and self-pairs
        overlap = sorted(token_sets[class1].intersection(token_sets[class2]))
        print(f"→ {class1} ↔ {class2} | Overlap: {len(overlap)}")
        if overlap:
            print("   ", ", ".join(overlap))
        else:
            print("    (No overlapping tokens)")
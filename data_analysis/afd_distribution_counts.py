from transformers import AutoTokenizer
from mamkit.data.datasets import MMUSEDFallacy, InputMode
from pathlib import Path
from collections import Counter
import matplotlib.pyplot as plt
import re
import nltk
from nltk.corpus import stopwords
from itertools import chain
import matplotlib.cm as cm
import numpy as np

# Download stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# BERT tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# Load dataset
TASK = 'afd'
loader = MMUSEDFallacy(
    task_name=TASK,
    input_mode=InputMode.TEXT_AUDIO,
    base_data_path=Path(__file__).parent.parent.resolve().joinpath('data'),
    with_context=False
)
split_info = next(loader.get_splits(key='mancini-et-al-2024'))
dataset_train = list(split_info.train)
dataset_val = list(split_info.val)
dataset_test = list(split_info.test)

dataset = [item for item in dataset_train] + [item for item in dataset_val] + [item for item in dataset_test]

# Filter fallacy-labeled texts
fallacy_texts = [ex[0] for ex in dataset if ex[2] == 1]

# Tokenize and filter tokens
all_tokens = []
for text in fallacy_texts:
    tokens = tokenizer.tokenize(text.lower())
    for tok in tokens:
        # Remove punctuation and stopwords
        if tok in stop_words: continue
        if re.fullmatch(r"[\W_]+", tok): continue  # punctuation
        if tok.isnumeric(): continue
        all_tokens.append(tok)

# Count top content tokens
token_counts = Counter(all_tokens)

# Print top 20
top_k = 37
print("\n🔍 Top Informative Tokens in Fallacy-Labeled Examples (after filtering):")
for token, count in token_counts.most_common(top_k):
    print(f"{token}: {count}")

# Plot
top_words = token_counts.most_common(top_k)
words, counts = zip(*top_words)


num_colors = len(words)
colors = [cm.viridis(i/num_colors) for i in range(num_colors)] 



# plt.figure(figsize=(10, 5))
# plt.bar(words, counts, color=colors)
# plt.title(f"Top {top_k} Content Tokens in Fallacy-Labeled Statements (AFD)")
# plt.xticks(rotation=45)
# plt.ylabel("Frequency")
# plt.tight_layout()
# plt.show()

plt.figure(figsize=(14, 7)) # Increased figure size for better readability with more elements
plt.bar(words, counts, color=colors)
plt.title(f"Top {top_k} Content Tokens in Fallacy-Labeled Statements (AFD)", fontsize=16)
plt.xticks(rotation=45, ha='right', fontsize=10) # 'ha' for horizontal alignment of rotated labels
plt.ylabel("Frequency", fontsize=12)
plt.yticks(fontsize=10)
plt.tight_layout(rect=[0, 0, 1, 0.95]) # Adjust layout to make space for title/stats

# --- Add counts on top of bars ---
for i, count in enumerate(counts):
    plt.text(i, count + 2, str(count), ha='center', va='bottom', fontsize=8, color='black') # Adjust y offset (+2) as needed

# --- Add statistical terms on top right ---
total_top_k_count = sum(counts)
average_frequency = np.mean(counts)
median_frequency = np.median(counts)
std_dev = np.std(counts)

stats_text = (
    f"Total Count (Top {top_k}): {total_top_k_count}\n"
    f"Avg. Freq: {average_frequency:.1f}\n"
    f"Median Freq: {median_frequency:.1f}\n"
    f"Std Dev: {std_dev:.1f}\n"
    f"Total Tokens: {len(token_counts)}\n"
    f"(Without Stopwords and Punctuation)"
)

# Position the text. (x, y) are in axes coordinates (0,0 is bottom-left, 1,1 is top-right)
plt.text(0.98, 0.98, stats_text, transform=plt.gca().transAxes,
         fontsize=10, verticalalignment='top', horizontalalignment='right',
         bbox=dict(boxstyle='round,pad=0.5', fc='white', alpha=0.7, ec='gray', lw=0.5))


plt.show()

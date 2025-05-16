import random
import logging
from pathlib import Path
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
import re
import nltk
from nltk.corpus import stopwords
import matplotlib.cm as cm
import numpy as np
from mamkit.data.datasets import MMUSEDFallacy, InputMode

# Download stopwords (only needs to run once)
try:
    nltk.data.find('corpora/stopwords')
except nltk.downloader.DownloadError:
    nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# BERT tokenizer (make sure transformers is installed: pip install transformers)
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# Label mapping (from dataset.py logic for AFC)
label_map = {
    0: 'Appeal to Emotion',
    1: 'Appeal to Authority',
    2: 'Ad Hominem',
    3: 'False Cause',
    4: 'Slippery Slope',
    5: 'Slogans'
}

loader = MMUSEDFallacy(
    task_name='afc',  # Ensure this is 'afc' for multiclass
    input_mode=InputMode.TEXT_AUDIO,
    base_data_path=Path(__file__).parent.parent.resolve().joinpath('data'),
    with_context=False
)
split_info = next(loader.get_splits(key='mancini-et-al-2024'))

dataset = list(split_info.train) + list(split_info.val) + list(split_info.test)

print(f"\nLoaded {len(dataset)} examples for task=AFC (all splits combined)")

# Compute label distribution for AFC
afc_label_counts = Counter([ex[2] for ex in dataset if ex[2] is not None])
print("\n📊 AFC Label Distribution:")
total_afc_labels = sum(afc_label_counts.values())

# Prepare data for plotting, ensuring labels are sorted by their integer key
sorted_labels_data = sorted(afc_label_counts.items(), key=lambda item: item[0])
afc_plot_labels_raw = [item[0] for item in sorted_labels_data]
afc_plot_labels_mapped = [label_map[item[0]] for item in sorted_labels_data]
afc_plot_counts = [item[1] for item in sorted_labels_data]
afc_plot_percentages = [(count / total_afc_labels) * 100 for count in afc_plot_counts]

for i, count in enumerate(afc_plot_counts):
    print(f"Label {afc_plot_labels_mapped[i]}: {count} ({afc_plot_percentages[i]:.2f}%)")

afc_colors = [
    '#87CEEB', # Sky Blue
    '#6495ED', # Cornflower Blue
    '#FFA07A', # Light Salmon (light orange/red)
    '#FF7F50', # Coral (medium orange)
    '#98FB98', # Pale Green (complementary)
    '#DA70D6'  # Orchid (another distinct complementary color)
]

plt.figure(figsize=(14, 7))
plt.bar(afc_plot_labels_mapped, afc_plot_counts, color=afc_colors)

plt.title("Class Label Distribution (AFC Task)", fontsize=16)
plt.xlabel("Fallacy Class", fontsize=12)
plt.ylabel("Number of Samples", fontsize=12)
plt.xticks(rotation=45, ha='right', fontsize=10)
plt.yticks(fontsize=10)

for i, count in enumerate(afc_plot_counts):
    count_y_offset = 100
    percent_y_offset = -100
    percent_text_color = 'white'

    if count < 500:
        count_y_offset = 50
        percent_y_offset = 0
        percent_text_color = 'black'

    plt.text(i, count + count_y_offset, f"{count}", ha='center', va='bottom', fontsize=9, color='black')
    plt.text(i, count + percent_y_offset, f"({afc_plot_percentages[i]:.2f}%)", ha='center', va='top', fontsize=8, color=percent_text_color, weight='bold')

# Add summary text to the top right
afc_summary_text = ""
for i, label_name in enumerate(afc_plot_labels_mapped):
    afc_summary_text += f"{label_name}: {afc_plot_counts[i]} ({afc_plot_percentages[i]:.2f}%)\n"
afc_summary_text = afc_summary_text.strip()

plt.text(0.98, 0.98, afc_summary_text, transform=plt.gca().transAxes,
         fontsize=9, verticalalignment='top', horizontalalignment='right',
         bbox=dict(boxstyle='round,pad=0.5', fc='white', alpha=0.7, ec='gray', lw=0.5))

plt.tight_layout()
plt.show()

tokens_per_class = defaultdict(list)
for text, _, label_idx, *_ in dataset:
    if text is None or label_idx is None:
        continue
    tokens = tokenizer.tokenize(text.lower())
    for tok in tokens:
        if tok in stop_words: continue
        if re.fullmatch(r"[\W_]+", tok): continue
        if tok.isnumeric(): continue
        tokens_per_class[label_idx].append(tok)

top_k = 20
fig, axs = plt.subplots(2, 3, figsize=(18, 12))
axs = axs.ravel()

for idx in range(6):
    current_label_name = label_map[idx]
    token_counts = Counter(tokens_per_class[idx])

    print(f"\n🔍 Top {top_k} tokens for label {idx}: {current_label_name}")
    top_items = token_counts.most_common(top_k)

    if not top_items:
        print(f"    No tokens found for {current_label_name} after filtering.")
        axs[idx].text(0.5, 0.5, "No data", horizontalalignment='center', verticalalignment='center', transform=axs[idx].transAxes, fontsize=12, color='gray')
        axs[idx].set_title(f"Top {top_k} Tokens - {current_label_name}", fontsize=12)
        axs[idx].tick_params(axis='x', rotation=45, ha='right', labelsize=9)
        axs[idx].tick_params(axis='y', labelsize=9)
        axs[idx].set_ylabel("Frequency", fontsize=10)
        continue

    words, counts = zip(*top_items)
    num_tokens_in_subplot = len(words)
    subplot_colors = [cm.viridis(i / num_tokens_in_subplot) for i in range(num_tokens_in_subplot)]

    axs[idx].bar(words, counts, color=subplot_colors)
    axs[idx].set_title(f"Top {top_k} Tokens - {current_label_name}", fontsize=12)
    axs[idx].tick_params(axis='x', rotation=45, labelsize=9)
    axs[idx].set_ylabel("Frequency", fontsize=10)
    axs[idx].tick_params(axis='y', labelsize=9)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.suptitle(f"Top {top_k} Most Frequent Tokens per Fallacy Class (AFC Task)", fontsize=18, y=0.99)
plt.show()
import random
import logging
from pathlib import Path
from collections import Counter
import matplotlib.pyplot as plt

from mamkit.data.datasets import MMUSEDFallacy, InputMode

logging.basicConfig(level=logging.INFO)

# Set this to 'afd' or 'afc' depending on what you want to analyze
TASK = 'afd'  # or 'afc'
SPLIT = 'train'  # or 'val' / 'test'

# Initialize loader
loader = MMUSEDFallacy(
    task_name=TASK,
    input_mode=InputMode.TEXT_AUDIO,
    base_data_path=Path(__file__).parent.parent.resolve().joinpath('data'),
    with_context=False
)

split_info = next(loader.get_splits(key='mancini-et-al-2024'))

dataset = {
    'train': split_info.train,
    'val': split_info.val,
    'test': split_info.test
}[SPLIT]

print(f"\nLoaded {len(dataset)} examples from {SPLIT} split for task={TASK}")

sampled = random.sample(list(dataset), k=100)

print("\n🔍 Sampled Examples (text + label):")
for i, ex in enumerate(sampled):
    text = ex[0]
    audio = ex[1]
    label = ex[2]
    context_text = ex[3]

    print(f"\n--- Example {i + 1} ---")
    print(f"Text      : {text}")
    print(f"Label     : {label}")
    print(f"Audio     : {audio}")
    if context_text:
        print(f"Context   : {context_text[:100]}...")  # preview first 100 chars

# Compute label distribution
label_counts = Counter([ex[2] for ex in dataset])
print("\n📊 Label Distribution:")
total = sum(label_counts.values())
for label, count in label_counts.items():
    print(f"Label {label}: {count} ({count/total:.2%})")


labels = [f"Label {label}" for label in label_counts.keys()]
counts = list(label_counts.values())
total = sum(counts)
percentages = [(count / total) * 100 for count in counts]
colors = ['#4287f5', '#f58a42']
plt.figure(figsize=(14, 7))
plt.bar(labels, counts, color=colors)

plt.title("Class Label Distribution (AFD Task)", fontsize=16)
plt.xlabel("Class Label", fontsize=12)
plt.ylabel("Number of Samples", fontsize=12)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)

for i, count in enumerate(counts):
    plt.text(i, count + 100, f"{count}", ha='center', va='bottom', fontsize=10, color='black')
    plt.text(i, count - 100, f"({percentages[i]:.2f}%)", ha='center', va='top', fontsize=9, color='white', weight='bold')

summary_text = (
    f"Label 0: {counts[0]} ({percentages[0]:.2f}%)\n"
    f"Label 1: {counts[1]} ({percentages[1]:.2f}%)"
)

plt.text(0.98, 0.98, summary_text, transform=plt.gca().transAxes,
         fontsize=10, verticalalignment='top', horizontalalignment='right',
         bbox=dict(boxstyle='round,pad=0.5', fc='white', alpha=0.7, ec='gray', lw=0.5))


plt.tight_layout()
plt.show()

import random
import logging
from pathlib import Path
from collections import Counter

from mamkit.data.datasets import MMUSEDFallacy, InputMode

logging.basicConfig(level=logging.INFO)

# Set this to 'afd' or 'afc' depending on what you want to analyze
TASK = 'afc'  # or 'afc'
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
        print(f"Context   : {context_text[:100]}...") 

# Compute label distribution
label_counts = Counter([ex[2] for ex in dataset])
print("\n📊 Label Distribution:")
total = sum(label_counts.values())
for label, count in label_counts.items():
    print(f"Label {label}: {count} ({count/total:.2%})")

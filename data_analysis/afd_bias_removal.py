import re
from pathlib import Path
from mamkit.data.datasets import MMUSEDFallacy, InputMode

# Define banned words (case-insensitive match)
banned_words = {
    "people", "american", "america", "president", "obama",
    "bush", "reagan", "mr", "uh", "campaign"
}

# Precompile regex for speed
banned_words_re = re.compile(r'\b(?:' + '|'.join(re.escape(w) for w in banned_words) + r')\b', re.IGNORECASE)

# Function to remove banned words from fallacy-labeled texts
def remove_bias_tokens(dataset, banned_tokens):
    new_examples = []
    for ex in dataset:
        text, audio, label, text_ctx, audio_ctx = ex
        if label == 1:
            words = text.split()
            filtered = [w for w in words if w.lower().strip(".,?!'\"") not in banned_tokens]
            new_text = " ".join(filtered)
            ex = (new_text, audio, label, text_ctx, audio_ctx)
        new_examples.append(ex)
    return new_examples

# Load dataset
loader = MMUSEDFallacy(
    task_name='afd',
    input_mode=InputMode.TEXT_AUDIO,
    base_data_path=Path(__file__).parent.parent.resolve().joinpath('data'),
    with_context=False
)
split_info = next(loader.get_splits(key='mancini-et-al-2024'))
original_dataset = list(split_info.train)

# Only fallacy-labeled examples that contain at least one banned word
to_check = [ex for ex in original_dataset if ex[2] == 1 and banned_words_re.search(ex[0])]
print(f"\nFound {len(to_check)} fallacy-labeled examples containing banned words.")

# Filter dataset
filtered_dataset = remove_bias_tokens(original_dataset, banned_words)

# Build map: original id -> filtered version
original_to_filtered = {id(orig): filt for orig, filt in zip(original_dataset, filtered_dataset)}

# Show up to 5 changed examples
print("\n🔍 Examples with changes:")
shown = 0
for orig in to_check:
    filtered = original_to_filtered.get(id(orig))
    if not filtered:
        continue
    if orig[0] != filtered[0]:
        print("\n---")
        print(f"Original : {orig[0]}")
        print(f"Filtered : {filtered[0]}")
        shown += 1
    if shown >= 5:
        break

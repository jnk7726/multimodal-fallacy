import matplotlib.pyplot as plt
import re

data_string = """
→ Appeal to Emotion ↔ Appeal to Authority | Overlap: 11
    american, believe, country, going, just, know, people, president, think, ve, want
→ Appeal to Emotion ↔ False Cause | Overlap: 6
    america, americans, people, president, ve, years
→ Appeal to Emotion ↔ Slogans | Overlap: 5
    america, going, make, president, want
→ Appeal to Emotion ↔ Ad Hominem | Overlap: 4
    just, know, people, president
→ Appeal to Emotion ↔ Slippery Slope | Overlap: 10
    america, american, going, people, president, states, think, united, ve, world
→ Appeal to Authority ↔ False Cause | Overlap: 3
    people, president, ve
→ Appeal to Authority ↔ Slogans | Overlap: 3
    going, president, want
→ Appeal to Authority ↔ Ad Hominem | Overlap: 8
    don, just, know, people, president, said, senator, uh
→ Appeal to Authority ↔ Slippery Slope | Overlap: 8
    american, going, people, president, said, think, uh, ve
→ False Cause ↔ Slogans | Overlap: 2
    america, president
→ False Cause ↔ Ad Hominem | Overlap: 3
    did, people, president
→ False Cause ↔ Slippery Slope | Overlap: 4
    america, people, president, ve
→ Slogans ↔ Ad Hominem | Overlap: 1
    president
→ Slogans ↔ Slippery Slope | Overlap: 5
    america, economy, going, ll, president
→ Ad Hominem ↔ Slippery Slope | Overlap: 4
    people, president, said, uh
"""

lines = data_string.strip().split('\n')
parsed_data = []
current_pair = {}

for line in lines:
    line = line.strip()
    if line.startswith('→'):

        if current_pair: 
            parsed_data.append(current_pair)
        
        # Regex to extract class names and overlap count
        match = re.match(r'→ (.*?) ↔ (.*?) \| Overlap: (\d+)', line)
        if match:
            current_pair = {
                'Class1': match.group(1).strip(),
                'Class2': match.group(2).strip(),
                'OverlapCount': int(match.group(3))
            }
        else:
            print(f"Warning: Could not parse header line: {line}")
    elif line:
        if current_pair:
            current_pair['OverlappingTokens'] = line.strip()
        else:
            print(f"Warning: Token line without preceding header: {line}")


if current_pair:
    parsed_data.append(current_pair)


table_data = []
header = ['Class 1', 'Class 2', 'Overlap Count', 'Overlapping Tokens']
table_data.append(header)

for item in parsed_data:
    table_data.append([
        item['Class1'],
        item['Class2'],
        item['OverlapCount'],
        item.get('OverlappingTokens', 'N/A') 
    ])


num_rows_in_table = len(table_data)
fig, ax = plt.subplots(figsize=(18, num_rows_in_table * 0.45)) 
ax.set_title('Overlapping TF-IDF Tokens Between All Class Pairs', fontsize=16, pad=20) 
ax.axis('off') 


table = ax.table(cellText=table_data, loc='center', cellLoc='left') # loc='center' centers the table in the figure


table.auto_set_font_size(False)
table.set_fontsize(10)



for (row, col), cell in table.get_celld().items():
    cell.set_edgecolor('black')
    if row == 0:
        cell.set_facecolor('#ADD8E6') 
        cell.set_text_props(weight='bold', color='black') 
    else: 
        cell.set_facecolor('white') 
        if row % 2 == 1:
            cell.set_facecolor('#F0F8FF')

table.auto_set_column_width(col=list(range(len(header))))

plt.tight_layout()

plt.savefig('token_overlap_table.png', bbox_inches='tight', dpi=300) # dpi for higher resolution

plt.show()
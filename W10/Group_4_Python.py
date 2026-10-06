from pathlib import Path

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

SHEET_NAME = 'Sheet1'
TITLE = 'Group 4'


def group4_rule(sepal_width):
    # Group 4 formula: =IF(B2>=3.4, "setosa", IF(B2<3.0, "versicolor", "virginica"))
    if sepal_width >= 3.4:
        return 'setosa'
    if sepal_width < 3.0:
        return 'versicolor'
    return 'virginica'


# 1. Read Excel file & drop empty formula rows
df = pd.read_excel(Path(__file__).parent / 'ANOVA.xlsx', sheet_name=SHEET_NAME)
df = df.dropna(subset=['species', 'sepal width (cm)']).copy()

# Apply the Group 4 rule directly to sepal width
df['Predicted_Species'] = df['sepal width (cm)'].apply(group4_rule)

# 2. Standardize text strings (lowercase & remove 'Iris-' prefix)
df['Actual'] = df['species'].astype(str).str.replace('Iris-', '').str.strip().str.lower()
df['Predicted'] = df['Predicted_Species'].astype(str).str.replace('Iris-', '').str.strip().str.lower()

# 3. Filter valid labels
labels = ['setosa', 'versicolor', 'virginica']
df = df[df['Actual'].isin(labels) & df['Predicted'].isin(labels)]

# 4. Compute Confusion Matrix & Accuracy
cm = confusion_matrix(df['Actual'], df['Predicted'], labels=labels)
acc = accuracy_score(df['Actual'], df['Predicted'])

# 5. Plot Heatmap
plt.figure(figsize=(7, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Pred Setosa', 'Pred Versicolor', 'Pred Virginica'],
            yticklabels=['Actual Setosa', 'Actual Versicolor', 'Actual Virginica'])

plt.title(f'{TITLE} Confusion Matrix\nOverall Accuracy: {acc*100:.2f}% ({len(df)} samples)')
plt.xlabel('Predicted Species')
plt.ylabel('Actual Species')
plt.tight_layout()
plt.show()

# 6. Output Statistical Classification Report
print("--- Classification Performance Report ---")
print(classification_report(df['Actual'], df['Predicted'], target_names=labels))
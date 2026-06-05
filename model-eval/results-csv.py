import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("/home/emre/Desktop/NATO/code/runs/detect/train19/results.csv")
df.columns = df.columns.str.strip()

# 2. Set up the plotting style for a clean, academic look
sns.set_theme(style="whitegrid")
plt.rcParams.update({'font.size': 12, 'font.family': 'serif'})

# 3. Create a 2x2 grid of subplots
fig, axs = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Model Training Performance Metrics', fontsize=16, fontweight='bold', y=0.98)

# --- Plot 1: Precision and Recall ---
axs[0, 0].plot(df['epoch'], df['metrics/precision(B)'], label='Precision', color='blue', linewidth=2)
axs[0, 0].plot(df['epoch'], df['metrics/recall(B)'], label='Recall', color='green', linewidth=2)
axs[0, 0].set_title('Precision & Recall', fontweight='bold')
axs[0, 0].set_xlabel('Epoch')
axs[0, 0].set_ylabel('Score')
axs[0, 0].set_ylim([0, 1.05])
axs[0, 0].legend(loc='lower right')

# --- Plot 2: mAP Scores ---
axs[0, 1].plot(df['epoch'], df['metrics/mAP50(B)'], label='mAP@0.5', color='purple', linewidth=2)
axs[0, 1].plot(df['epoch'], df['metrics/mAP50-95(B)'], label='mAP@0.5:0.95', color='orange', linewidth=2)
axs[0, 1].set_title('Mean Average Precision (mAP)', fontweight='bold')
axs[0, 1].set_xlabel('Epoch')
axs[0, 1].set_ylabel('Score')
axs[0, 1].set_ylim([0, 1.05])
axs[0, 1].legend(loc='lower right')

# --- Plot 3: Box Loss (Train vs Val) ---
axs[1, 0].plot(df['epoch'], df['train/box_loss'], label='Train Box Loss', color='blue', linestyle='--')
axs[1, 0].plot(df['epoch'], df['val/box_loss'], label='Val Box Loss', color='red', linewidth=2)
axs[1, 0].set_title('Bounding Box Loss', fontweight='bold')
axs[1, 0].set_xlabel('Epoch')
axs[1, 0].set_ylabel('Loss')
axs[1, 0].legend(loc='upper right')

# --- Plot 4: Classification Loss (Train vs Val) ---
axs[1, 1].plot(df['epoch'], df['train/cls_loss'], label='Train Cls Loss', color='blue', linestyle='--')
axs[1, 1].plot(df['epoch'], df['val/cls_loss'], label='Val Cls Loss', color='red', linewidth=2)
axs[1, 1].set_title('Classification Loss', fontweight='bold')
axs[1, 1].set_xlabel('Epoch')
axs[1, 1].set_ylabel('Loss')
axs[1, 1].legend(loc='upper right')

# 4. Adjust layout and save the figure
plt.tight_layout()
plt.subplots_adjust(top=0.92) # Leave room for the main title

# Save as a high-resolution PNG for your paper
plt.savefig('training_metrics_plot.png', dpi=300, bbox_inches='tight')

# Display the plot if running in a notebook or interactive window
plt.show()
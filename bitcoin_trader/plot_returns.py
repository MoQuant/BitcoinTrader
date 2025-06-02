import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
import datetime

dt = lambda u: datetime.datetime.fromtimestamp(u).strftime('%m-%d-%Y %I:%M:%S')

# Set Seaborn theme for better aesthetics
sns.set_theme(style="darkgrid")

# Load and preprocess data
with open('TradeLog.csv', 'r') as file:
    data = np.array([[float(j) for j in i.strip().split(',')] for i in file.readlines()])

returns = data[:, 1]
stamps = list(map(dt, data[:, 0]))

lg = 'limegreen'

# Create figure and subplots
fig, (ax, ay) = plt.subplots(2, 1, figsize=(12, 7), gridspec_kw={'height_ratios': [2, 1]})
fig.suptitle('Trade Return Analysis', fontsize=18, fontweight='bold', color=lg)
fig.patch.set_facecolor('black')

# Histogram with KDE
sns.histplot(returns, bins=30, kde=True, color='mediumslateblue', edgecolor='black', ax=ax)
ax.set_title('Distribution of Returns', fontsize=14, color=lg)
ax.set_xlabel('Returns', color=lg)
ax.set_ylabel('Frequency', color=lg)
mean_return = np.mean(returns)
ax.axvline(mean_return, color='red', linestyle='--', linewidth=2)
ax.text(mean_return, ax.get_ylim()[1]*0.9, f'Mean: {mean_return:.4f}', color='red', ha='center', fontsize=10)
ax.tick_params('x', colors=lg)
ax.tick_params('y', colors=lg)

# Time series plot
ay.plot(stamps, returns, color='tomato', linewidth=1.5)
ay.set_title('Returns Over Time', fontsize=14, color=lg)
ay.set_xlabel('Timestamp', color=lg)
ay.set_ylabel('Returns', color=lg)
ay.set_xticklabels(ay.get_xticklabels(), rotation=45)
ay.tick_params('x', colors=lg)
ay.tick_params('y', colors=lg)


# Tight layout to avoid overlap
plt.tight_layout(rect=[0, 0, 1, 0.95])  # Leave space for suptitle

plt.show()

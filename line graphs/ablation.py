import matplotlib.pyplot as plt
import numpy as np

# # Data
pruning_rates = [0, 10, 20, 30, 40, 50]
# one_shot_miou = [42.15, 43.90, 44.80, 46.59, 44.20, 42.80]
# five_shot_miou = [47.10, 48.20, 48.91, 49.72, 47.90, 46.50]

# Bar width and positions
bar_width = 0.35
x = np.arange(len(pruning_rates))

# # Plotting
# plt.figure(figsize=(10, 6))
# plt.bar(x - bar_width / 2, one_shot_miou, bar_width, label="1-Shot mIoU", color="skyblue")
# plt.bar(x + bar_width / 2, five_shot_miou, bar_width, label="5-Shot mIoU", color="salmon")

# # Adding labels and title
# plt.xlabel("Pruning Rate (%)", fontsize=12)
# plt.ylabel("mIoU (%)", fontsize=12)
# #plt.title("mIoU Performance at Different Pruning Rates", fontsize=14)
# plt.xticks(x, pruning_rates)
# plt.legend(fontsize=12)
# plt.grid(axis="y", linestyle="--", alpha=0.7)
# plt.tight_layout()

# # Show the plot
# plt.show()

# ISAID

one_shot_miou_updated = [35.67, 35.86, 36.53, 37.05, 33.76, 32.02]
five_shot_miou_updated = [41.07, 41.85, 42.91, 43.90, 37.57, 36.34]

# Plotting
plt.figure(figsize=(10, 6))
plt.bar(x - bar_width / 2, one_shot_miou_updated, bar_width, label="1-Shot mIoU", color="skyblue")
plt.bar(x + bar_width / 2, five_shot_miou_updated, bar_width, label="5-Shot mIoU", color="salmon")

# Adding labels and title
plt.xlabel("Pruning Rate (%)", fontsize=12)
plt.ylabel("mIoU (%)", fontsize=12)
#plt.title("Updated mIoU Performance at Different Pruning Rates", fontsize=14)
plt.xticks(x, pruning_rates)
plt.legend(fontsize=12)
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.tight_layout()

# Show the plot
plt.show()

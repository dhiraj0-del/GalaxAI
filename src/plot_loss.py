import json
import matplotlib.pyplot as plt

HISTORY_PATH = "results/model1_v2_history.json"
OUTPUT_PATH = "results/model1_v2_loss_breakdown.png"

BINARY_TASKS = [
    "featured",
    "edge_on",
    "bar",
    "spiral_arms",
    "disturbed",
    "merger",
    "clumpy",
    "symmetry"
]

MULTICLASS_TASKS = [
    "bulge",
    "roundedness"
]

# Load history
with open(HISTORY_PATH, "r") as f:
    history = json.load(f)

epochs = [record["epoch"] for record in history]

binary_losses = []
multiclass_losses = []
total_losses = []

for record in history:

    task_losses = record["validation"]["task_losses"]

    # Sum BCE losses
    binary_loss = sum(
        task_losses[task]
        for task in BINARY_TASKS
    )

    # Sum Cross-Entropy losses
    multiclass_loss = sum(
        task_losses[task]
        for task in MULTICLASS_TASKS
    )

    binary_losses.append(binary_loss)
    multiclass_losses.append(multiclass_loss)
    total_losses.append(binary_loss + multiclass_loss)

# Find best validation epoch
best_index = total_losses.index(min(total_losses))
best_epoch = epochs[best_index]
best_total_loss = total_losses[best_index]

# Plot
plt.figure(figsize=(11, 6))

plt.plot(
    epochs,
    binary_losses,
    linewidth=2,
    label="Binary BCE Loss"
)

plt.plot(
    epochs,
    multiclass_losses,
    linewidth=2,
    label="Multiclass Cross-Entropy Loss"
)

plt.plot(
    epochs,
    total_losses,
    linewidth=2.5,
    label="Total Validation Loss"
)

plt.scatter(
    best_epoch,
    best_total_loss,
    s=100,
    zorder=5,
    label=f"Best Epoch ({best_epoch})"
)

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("GALAXAI Model 1 V2 — Loss Breakdown")

plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.savefig(
    OUTPUT_PATH,
    dpi=300,
    bbox_inches="tight"
)

print("=" * 60)
print("LOSS BREAKDOWN")
print("=" * 60)
print(f"Best Epoch: {best_epoch}")
print(f"Best Total Validation Loss: {best_total_loss:.6f}")
print(f"Binary BCE Loss at Best Epoch: {binary_losses[best_index]:.6f}")
print(
    f"Multiclass Cross-Entropy Loss at Best Epoch: "
    f"{multiclass_losses[best_index]:.6f}"
)
print(f"Graph saved to: {OUTPUT_PATH}")
print("=" * 60)

plt.show()
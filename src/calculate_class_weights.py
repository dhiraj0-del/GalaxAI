import pandas as pd
import json
from pathlib import Path

TRAIN_CSV = Path("results/multitask_train.csv")
OUTPUT_JSON = Path("results/class_weights.json")

BINARY_TASKS = [
    "featured",
    "edge_on",
    "bar",
    "spiral_arms",
    "disturbed",
    "merger",
    "clumpy",
    "symmetry",
]

MULTICLASS_TASKS = {
    "bulge": 4,
    "roundedness": 3,
}


def binary_weights(df, task):
    target_col = task
    mask_col = f"{task}_mask"

    valid = df[df[mask_col] == 1]

    positives = (valid[target_col] == 1).sum()
    negatives = (valid[target_col] == 0).sum()

    pos_weight = negatives / positives if positives > 0 else 1.0

    return {
        "negative": int(negatives),
        "positive": int(positives),
        "pos_weight": round(float(pos_weight), 6),
    }


def multiclass_weights(df, task, num_classes):
    target_col = task
    mask_col = f"{task}_mask"

    valid = df[df[mask_col] == 1]

    counts = []

    for cls in range(num_classes):
        counts.append(int((valid[target_col] == cls).sum()))

    total = sum(counts)

    weights = []

    for count in counts:
        if count > 0:
            weights.append(
                total / (num_classes * count)
            )
        else:
            weights.append(1.0)

    return {
        "class_counts": counts,
        "class_weights": [
            round(float(w), 6)
            for w in weights
        ],
    }


def main():

    print("=" * 75)
    print("GALAXAI — TRAINING CLASS WEIGHTS")
    print("=" * 75)

    df = pd.read_csv(TRAIN_CSV)

    print(f"\nTraining samples: {len(df)}")

    result = {
        "binary": {},
        "multiclass": {},
    }

    print("\nBINARY TASKS")
    print("-" * 75)

    for task in BINARY_TASKS:

        info = binary_weights(df, task)

        result["binary"][task] = info

        print(
            f"{task:15s} "
            f"negative={info['negative']:6d}  "
            f"positive={info['positive']:6d}  "
            f"pos_weight={info['pos_weight']:.4f}"
        )

    print("\nMULTICLASS TASKS")
    print("-" * 75)

    for task, num_classes in MULTICLASS_TASKS.items():

        info = multiclass_weights(
            df,
            task,
            num_classes
        )

        result["multiclass"][task] = info

        print(f"\n{task}:")
        print(f"  class counts : {info['class_counts']}")
        print(f"  class weights: {info['class_weights']}")

    OUTPUT_JSON.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(OUTPUT_JSON, "w") as f:
        json.dump(
            result,
            f,
            indent=2
        )

    print("\n" + "=" * 75)
    print(f"Saved: {OUTPUT_JSON}")
    print("=" * 75)


if __name__ == "__main__":
    main()
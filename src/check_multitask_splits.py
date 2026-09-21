import pandas as pd

files = {
    "TRAIN": "results/multitask_train.csv",
    "VALIDATION": "results/multitask_validation.csv",
    "TEST": "results/multitask_test.csv",
}

targets = [
    "featured",
    "edge_on",
    "bar",
    "spiral_arms",
    "disturbed",
    "merger",
    "clumpy",
    "symmetry",
]

for split_name, path in files.items():

    df = pd.read_csv(path)

    print("=" * 75)
    print(f"{split_name}: {len(df):,} images")
    print("=" * 75)

    for target in targets:

        mask = df[f"{target}_mask"] == 1

        positive = ((df[target] == 1) & mask).sum()
        negative = ((df[target] == 0) & mask).sum()
        valid = mask.sum()

        print(
            f"{target:15s} "
            f"Positive: {positive:6,} | "
            f"Negative: {negative:6,} | "
            f"Valid: {valid:6,}"
        )

    print()
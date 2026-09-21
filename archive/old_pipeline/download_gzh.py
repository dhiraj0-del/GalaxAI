from galaxy_datasets import gz_hubble

data_root = r"D:\GalaxAI\data\gzh"

catalog, label_cols = gz_hubble(
    root=data_root,
    train=True,
    download=True
)

print("Download complete!")
print("Number of galaxies:", len(catalog))
print("Number of label columns:", len(label_cols))

print("\nLabel columns:")
for col in label_cols:
    print(col)

print("\nCatalog columns:")
print(catalog.columns.tolist())
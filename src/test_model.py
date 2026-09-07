import torch

from models import CustomCNN


# =========================================================
# SETTINGS
# =========================================================

BATCH_SIZE = 4
IMAGE_SIZE = 224
NUM_CLASSES = 4


# =========================================================
# CREATE MODEL
# =========================================================

print("=" * 70)
print("CUSTOM CNN - MODEL TEST")
print("=" * 70)

model = CustomCNN(num_classes=NUM_CLASSES)

print("\nModel created successfully.")


# =========================================================
# COUNT PARAMETERS
# =========================================================

total_parameters = sum(
    parameter.numel()
    for parameter in model.parameters()
)

trainable_parameters = sum(
    parameter.numel()
    for parameter in model.parameters()
    if parameter.requires_grad
)

print(f"\nTotal parameters     : {total_parameters:,}")
print(f"Trainable parameters : {trainable_parameters:,}")


# =========================================================
# TEST FORWARD PASS
# =========================================================

print("\nTesting forward pass...")

dummy_input = torch.randn(
    BATCH_SIZE,
    3,
    IMAGE_SIZE,
    IMAGE_SIZE
)

output = model(dummy_input)

print(f"Input shape  : {dummy_input.shape}")
print(f"Output shape : {output.shape}")

print(f"\nOutput:")
print(output)


# =========================================================
# VERIFY OUTPUT
# =========================================================

expected_shape = (BATCH_SIZE, NUM_CLASSES)

if tuple(output.shape) == expected_shape:

    print("\nPASS: Model produces exactly 4 outputs.")

else:

    print(
        f"\nERROR: Expected {expected_shape}, "
        f"but received {tuple(output.shape)}"
    )


# =========================================================
# FINAL
# =========================================================

print("\n" + "=" * 70)
print("CUSTOM CNN TEST COMPLETE")
print("=" * 70)
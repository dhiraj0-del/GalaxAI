import torch

from models import GalaxyCNNV2


def main():

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("=" * 70)
    print("GALAXAI CNN V2 — ARCHITECTURE TEST")
    print("=" * 70)

    print(f"Device: {device}")

    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    model = GalaxyCNNV2().to(device)

    model.eval()

   
    # Parameter count


    total_params = sum(
        p.numel()
        for p in model.parameters()
    )

    trainable_params = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print(f"\nTotal parameters:     {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")


    # Test input


    x = torch.randn(
        4,
        3,
        224,
        224,
        device=device,
    )

    print(f"\nInput shape: {tuple(x.shape)}")

  
    # Forward pass


    with torch.no_grad():

        outputs = model(x)

    print("\nOUTPUTS")
    print("-" * 70)

    expected_shapes = {
        "featured": (4,),
        "edge_on": (4,),
        "bar": (4,),
        "spiral_arms": (4,),
        "bulge": (4, 4),
        "roundedness": (4, 3),
        "disturbed": (4,),
        "merger": (4,),
        "clumpy": (4,),
        "symmetry": (4,),
    }

    for name, output in outputs.items():

        expected = expected_shapes[name]

        print(
            f"{name:15s} "
            f"{tuple(output.shape)} "
            f"{'PASS' if tuple(output.shape) == expected else 'FAIL'}"
        )

    
    # Total logits
   

    total_logits = sum(
        output.shape[1] if output.ndim == 2 else 1
        for output in outputs.values()
    )

    print(f"\nTotal morphology tasks: 10")
    print(f"Total output logits:    {total_logits}")

    # Final validation


    assert len(outputs) == 10

    for name, output in outputs.items():
        assert tuple(output.shape) == expected_shapes[name]

    assert total_logits == 15

    print("\n" + "=" * 70)
    print("MODEL V2 TEST: PASS")
    print("=" * 70)


if __name__ == "__main__":
    main()
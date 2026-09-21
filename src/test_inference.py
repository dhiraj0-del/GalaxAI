from PIL import Image

from inference import load_model, predict, model_info


IMAGE_PATH = "D:/Dhiraj/GalaxAI/data/gzh/images/10003080.jpg"


def main():

    print("=" * 70)
    print("GALAXAI — INFERENCE TEST")
    print("=" * 70)

    print("\nModel information:")

    for key, value in model_info().items():
        print(f"  {key}: {value}")

    print("\nLoading model...")

    model = load_model()

    print("Model loaded successfully.")

    image = Image.open(IMAGE_PATH)

    print(
        f"\nImage: {image.size} "
        f"{image.mode}"
    )

    results = predict(
        image,
        model,
    )

    print("\n" + "=" * 70)
    print("PREDICTIONS")
    print("=" * 70)

    for task, result in results.items():

        if "detected" in result:

            status = (
                "DETECTED"
                if result["detected"]
                else "NOT DETECTED"
            )

            print(
                f"{result['label']:25s} "
                f"{status:12s} "
                f"score={result['score']:.4f}"
            )

        else:

            print(
                f"{result['label']:25s} "
                f"{result['prediction']:20s} "
                f"score={result['score']:.4f}"
            )

    print("\n" + "=" * 70)
    print("INFERENCE TEST: PASS")
    print("=" * 70)


if __name__ == "__main__":
    main()
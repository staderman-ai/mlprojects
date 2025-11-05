import argparse
from functools import lru_cache
from pathlib import Path
from typing import Optional, Tuple

from PIL import Image, UnidentifiedImageError
from transformers import BlipForConditionalGeneration, BlipProcessor

SUPPORTED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".gif",
    ".webp",
}

EXIT_COMMANDS = {"exit", "quit", "q"}


@lru_cache(maxsize=2)
def load_model(device: Optional[str] = None) -> Tuple[BlipProcessor, BlipForConditionalGeneration]:
    """Load the BLIP captioning model, optionally placing it on a specific device."""
    model_name = "Salesforce/blip-image-captioning-base"
    processor = BlipProcessor.from_pretrained(model_name)
    model = BlipForConditionalGeneration.from_pretrained(model_name)

    if device:
        model.to(device)

    return processor, model


def can_access_image(image_path: Path) -> bool:
    """Return True when the path exists, has a supported suffix, and can be opened."""
    if not image_path.is_file():
        return False

    if image_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        return False

    try:
        with Image.open(image_path) as image:
            image.verify()
    except (UnidentifiedImageError, OSError):
        return False

    return True


def describe_image(
    image_path: Path,
    device: Optional[str] = None,
    max_tokens: int = 30,
    *,
    processor: Optional[BlipProcessor] = None,
    model: Optional[BlipForConditionalGeneration] = None,
) -> str:
    """Generate a caption for the given image or report failure."""
    if not can_access_image(image_path):
        return "unable to read the file"

    if processor is None or model is None:
        processor, model = load_model(device)
    elif device:
        # Ensure the cached model is on the requested device when specified at call time.
        model.to(device)

    try:
        with Image.open(image_path) as image:
            rgb_image = image.convert("RGB")
    except (UnidentifiedImageError, OSError):
        return "unable to read the file"

    inputs = processor(rgb_image, return_tensors="pt").to(model.device)
    output_ids = model.generate(**inputs, max_new_tokens=max_tokens)
    return processor.decode(output_ids[0], skip_special_tokens=True)


def run_service(device: Optional[str], max_tokens: int) -> None:
    """Keep accepting image paths, captioning each until the user exits."""
    print("Loading captioning model (first run may take a minute)...")
    processor, model = load_model(device)
    print(
        "Captioning service ready. Enter an image path or type 'exit' to quit."
    )

    while True:
        try:
            raw_input_path = input("image> ").strip()
        except EOFError:
            print()
            break

        if not raw_input_path:
            continue

        if raw_input_path.lower() in EXIT_COMMANDS:
            break

        caption = describe_image(
            Path(raw_input_path),
            max_tokens=max_tokens,
            processor=processor,
            model=model,
        )
        print(caption)

    print("Service stopped.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Describe the contents of an image.")
    parser.add_argument(
        "image",
        type=Path,
        nargs="?",
        help="Optional path to caption once instead of starting the service.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Optional device override (e.g. 'cuda', 'cuda:0', or 'cpu')",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=30,
        help="Maximum number of tokens to generate for each caption (default: 30)",
    )
    parser.add_argument(
        "--serve",
        action="store_true",
        help="Start the interactive service even if an image path is provided.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.image and not args.serve:
        caption = describe_image(args.image, args.device, args.max_tokens)
        print(caption)
        return

    run_service(args.device, args.max_tokens)


if __name__ == "__main__":
    main()
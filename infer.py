import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Generate initial CME masks using SAM ViT-B and LoRA.")
    parser.add_argument("--input", default="./processed_png")
    parser.add_argument("--output", default="./predicted_masks")
    parser.add_argument("--checkpoint", default="./sam_vit_b_01ec64.pth")
    parser.add_argument("--lora", default="./lora_weights/lora_rank512.safetensors")
    args = parser.parse_args()
    for path in (args.checkpoint, args.lora):
        if not Path(path).is_file():
            parser.error("Missing checkpoint: " + path)
    if not Path(args.input).is_dir():
        parser.error("Input directory does not exist")

    import torch
    import numpy as np
    from PIL import Image
    from src.segment_anything import build_sam_vit_b, SamPredictor
    from src.lora import LoRA_sam

    device = "cuda" if torch.cuda.is_available() else "cpu"
    sam = build_sam_vit_b(checkpoint=args.checkpoint)
    sam_lora = LoRA_sam(sam, 512)
    sam_lora.load_lora_parameters(args.lora)
    predictor = SamPredictor(sam_lora.sam.to(device).eval())
    input_root, output_root = Path(args.input), Path(args.output)
    count = 0
    for image_path in sorted(input_root.rglob("*")):
        if image_path.suffix.lower() != ".png" or "masks" in str(image_path.parent).lower():
            continue
        with Image.open(image_path) as source:
            if source.size != (1024, 1024):
                raise ValueError(f"Expected 1024 x 1024 input: {image_path}")
            image = np.array(source.convert("RGB"))
        predictor.set_image(image)
        masks, _, _ = predictor.predict(box=np.array([0, 0, 1024, 1024]), multimask_output=False)
        name = "mask_" + image_path.name[len("running_"):] if image_path.name.startswith("running_") else image_path.name
        target = output_root / image_path.relative_to(input_root).parent / name
        target.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(masks[0].astype(np.uint8) * 255).save(target)
        count += 1
        print(f"Saved mask: {target}")
    if not count:
        raise ValueError("No eligible PNG images found")


if __name__ == "__main__":
    main()

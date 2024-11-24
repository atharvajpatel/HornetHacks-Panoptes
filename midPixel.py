import torch
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from transformers import SamModel, SamProcessor
from tqdm import tqdm
import torch.cuda.amp as amp

def segment_image(input_image_path, output_image_path, model_name="facebook/sam-vit-base"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load the image with progress bar
    with tqdm(total=4, desc="Processing") as pbar:
        pbar.set_description("Loading image")
        image = Image.open(input_image_path).convert("RGB")
        pbar.update(1)

        # Load the SAM model and processor
        pbar.set_description("Loading model")
        processor = SamProcessor.from_pretrained(model_name)
        model = SamModel.from_pretrained(model_name).to(device)
        model.eval()  # Set model to evaluation mode
        pbar.update(1)

        # Prepare the image for the model
        pbar.set_description("Processing image")
        inputs = processor(images=image, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}  # Move all inputs to device
        pbar.update(1)

        # Perform inference with mixed precision
        with torch.no_grad(), amp.autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu'):
            outputs = model(**inputs)

        # Process the output masks
        pbar.set_description("Generating output")
        masks = processor.image_processor.post_process_masks(
            outputs.pred_masks.cpu(),
            inputs["original_sizes"].cpu(),
            inputs["reshaped_input_sizes"].cpu()
        )

        # Convert the first mask to a binary mask
        mask = masks[0][0].numpy()
        binary_mask = (mask > 0.5).astype(np.uint8) * 255

        # Overlay the mask on the original image
        plt.figure(figsize=(10, 10))
        plt.imshow(image)
        plt.imshow(binary_mask, alpha=0.5, cmap='jet')
        plt.axis('off')

        # Save the output image
        plt.savefig(output_image_path, bbox_inches='tight', pad_inches=0)
        plt.close()
        pbar.update(1)

    # Clear CUDA cache after processing
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

if __name__ == "__main__":
    input_image_path = "army-bottle.jpg"
    output_image_path = "SAM-army-bottle.jpg"

    # Enable deterministic behavior for reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    segment_image(input_image_path, output_image_path)
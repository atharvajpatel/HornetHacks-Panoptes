import torch
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from transformers import SamModel, SamProcessor
from tqdm import tqdm
from torch import amp
from skimage.transform import resize

def segment_image(input_image_path, output_image_path, model_name="facebook/sam-vit-base"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load the image with progress bar
    with tqdm(total=4, desc="Processing") as pbar:
        pbar.set_description("Loading image")
        image = Image.open(input_image_path).convert("RGB")
        # Convert image to numpy array
        image_np = np.array(image)
        print(f"Image shape: {image_np.shape}")
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
        pbar.set_description("Performing inference")
        with torch.no_grad(), amp.autocast(device_type=device.type):
            outputs = model(**inputs)

        # Process the output masks
        masks = processor.image_processor.post_process_masks(
            outputs.pred_masks.cpu(),
            inputs["original_sizes"].cpu(),
            inputs["reshaped_input_sizes"].cpu()
        )

        # Convert the first mask to a binary mask
        mask = masks[0][0].numpy()
        print(f"Initial mask shape: {mask.shape}")
        binary_mask = (mask > 0.5).astype(np.uint8) * 255

        # Reshape mask if needed
        if binary_mask.ndim == 3:
            binary_mask = binary_mask[0]  # Take the first channel if it's 3D
        
        # Ensure the mask has the same height and width as the input image
        if binary_mask.shape != image_np.shape[:2]:
            print(f"Resizing mask from {binary_mask.shape} to {image_np.shape[:2]}")
            
            binary_mask = resize(binary_mask, image_np.shape[:2], order=0, preserve_range=True).astype(np.uint8)

        # Create a color overlay
        overlay = np.zeros((*image_np.shape[:2], 4))  # RGBA
        overlay[binary_mask > 0] = [1, 0, 0, 0.5]  # Red with 0.5 opacity where mask is active

        # Create the visualization
        plt.figure(figsize=(10, 10))
        plt.imshow(image_np)  # Original image
        plt.imshow(overlay, alpha=0.5)  # Overlay mask
        plt.axis('off')

        # Save the output image
        plt.savefig(output_image_path, bbox_inches='tight', pad_inches=0, dpi=300)
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
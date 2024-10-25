from diffusers import StableDiffusionPipeline
import torch
import torch.nn.utils.prune as prune
from torch.utils.data import DataLoader, Dataset
import numpy as np
import cv2  # OpenCV for efficient image saving
import time

# Load Stable Diffusion with mixed precision (FP16) for efficient GPU usage
pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", torch_dtype=torch.float16)
pipe = pipe.to("cuda")

# Function to prune the unet model in Stable Diffusion pipeline
def prune_model(unet, amount=0.2):
    for module in unet.modules():
        if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear):
            prune.l1_unstructured(module, name="weight", amount=amount)
    return unet

# Prune the UNet model in Stable Diffusion pipeline
pipe.unet = prune_model(pipe.unet, amount=0.2)

# Check if CUDA is available
if torch.cuda.is_available():
    device = torch.device("cuda")  # Use the GPU
    print("CUDA is available, using GPU")
else:
    device = torch.device("cpu")  # Fall back to the CPU
    print("CUDA is not available, using CPU")
print(torch.cuda.get_device_name(0))  # Displays the name of the GPU

# Custom Dataset class to handle prompt inputs
class PromptDataset(Dataset):
    def __init__(self, prompts):
        self.prompts = prompts

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        return self.prompts[idx]

# Function to generate and save images in batch
def generate_images(prompts, batch_size=2):
    prompt_dataset = PromptDataset(prompts)

    # Set num_workers=0 to avoid multiprocessing issues
    prompt_loader = DataLoader(prompt_dataset, batch_size=batch_size, num_workers=0, pin_memory=True)

    for batch_num, batch in enumerate(prompt_loader):
        start_time = time.time()  # Track batch generation time
        images = pipe(batch).images  # Generate a batch of images

        # Save images using OpenCV for efficient processing
        for i, image in enumerate(images):
            img_num = batch_num * batch_size + i + 1
            cv_image = np.array(image)[:, :, ::-1]  # Convert RGB to BGR for OpenCV
            cv2.imwrite(f"optimized_image_{img_num}.png", cv_image)

        print(f"Batch {batch_num+1} generated in {time.time() - start_time:.2f} seconds.")

# User input for prompts and batch size
prompts = [
    "a futuristic cityscape at sunset",
    "a mountain landscape with clouds",
    "a serene lake at dawn with reflections",
    "a robot walking through a neon-lit street"
]
batch_size = 2

# Generate and save images
generate_images(prompts, batch_size)


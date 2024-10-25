# Text-to-Image Generation Using Stable Diffusion
The goal of this project is to use Stable Diffusion, a state-of-the-art text-to-image model, to generate images from descriptive text prompts. The project demonstrates how deep learning models can be used to generate high-quality images based on natural language input, automating creative tasks and exploring the possibilities of AI in art generation.

## Project Overview
This project uses **Stable Diffusion** to generate images from text prompts. Users can input descriptive text, and the model generates high-quality images based on the input.

## Objective
Demonstrate the potential of AI for creative tasks like image generation using deep learning models.

## Features
- Generate images from text prompts.
- Support for generating multiple images at once.
- Automatic saving of images with numbered filenames.
- Utilizes GPU acceleration via CUDA for faster image generation.

# Optimization Steps

1. **Model Pruning & Mixed Precision**:  
   - Reduced GPU memory usage by pruning less important weights in the `UNet` model and using FP16 precision instead of FP32. This optimization allowed better GPU utilization and faster image processing.

2. **Batch Processing & OpenCV for Efficient Saving**:  
   - Leveraged `DataLoader` to handle multiple prompts in parallel, reducing processing time per image. Used OpenCV for image saving to improve speed and efficiency when writing files to disk.

3. **Error Handling with DataLoader**:  
   - Addressed `DataLoader` worker crashes by setting `num_workers=0`, which avoids multiprocessing conflicts on GPU and improves stability. Enabled `pin_memory=True` for faster data transfer between CPU and GPU.

4. **Adjusting Batch Size**:  
   - Modified batch size to prevent GPU memory overflow, ensuring smooth processing even on limited hardware.


## How I Achieved the Goal
1. Installed **PyTorch** with CUDA support to ensure the GPU could be used for fast image generation.
2. Loaded the **Stable Diffusion** model using the `diffusers` library.
3. Implemented a loop to generate multiple images from a single text prompt.
4. Automatically saved each generated image with a numbered file name for easy identification.

## Challenges
- **CUDA Compatibility**: Ensuring that PyTorch was properly compiled with CUDA support to utilize the GPU for efficient image generation.
- **Resource Management**: Handling the large size of the Stable Diffusion model and managing system memory and GPU resources effectively to avoid crashes or performance issues.
- **Multiple Image Generation**: Designing a system that can generate multiple images in sequence and save each with a unique name while ensuring the process remains efficient.

## How Stable Diffusion Helped
Stable Diffusion provided a robust and efficient model for generating images from text. Its ability to interpret a wide range of prompts allowed for flexible and creative outputs. By leveraging the pre-trained weights from Stable Diffusion, I was able to:
- Produce high-resolution, detailed images.
- Generate images quickly using GPU acceleration.
- Explore different prompts and produce multiple images in batch mode.

## How to Run the Project
1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/text-to-image-stable-diffusion.git
   cd text-to-image-stable-diffusion

2. Install the required dependencies

To run this project, you'll need the following Python packages:

- `torch` (version >= 1.9.0) - PyTorch framework for deep learning.
- `torchvision` (version >= 0.10.0) - For image processing utilities.
- `diffusers` (version >= 0.3.0) - Library for running diffusion models.
- `numpy` - For numerical operations and array manipulations.
- `opencv-python` - For efficient image reading and writing.
- `pillow` - For basic image handling.

You can install these dependencies using pip. Here’s the command to install them:

   ```bash
   pip install torch torchvision diffusers numpy opencv-python pillow





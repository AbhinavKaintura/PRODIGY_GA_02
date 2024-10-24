# Text-to-Image Generation Using Stable Diffusion
The goal of this project is to use Stable Diffusion, a state-of-the-art text-to-image model, to generate images from descriptive text prompts. The project demonstrates how deep learning models can be used to generate high-quality images based on natural language input, automating creative tasks and exploring the possibilities of AI in art generation.

## Project Overview
This project uses **Stable Diffusion** to generate images from text prompts. Users can input descriptive text, and the model generates high-quality images based on the input.

## Features
- Generate images from text prompts.
- Support for generating multiple images at once.
- Automatic saving of images with numbered filenames.
- Utilizes GPU acceleration via CUDA for faster image generation.

## Objective
Demonstrate the potential of AI for creative tasks like image generation using deep learning models.

## Challenges
- **CUDA Compatibility**: Ensuring that PyTorch was properly compiled with CUDA support to utilize the GPU for efficient image generation.
- **Resource Management**: Handling the large size of the Stable Diffusion model and managing system memory and GPU resources effectively to avoid crashes or performance issues.
- **Multiple Image Generation**: Designing a system that can generate multiple images in sequence and save each with a unique name while ensuring the process remains efficient.

## How I Achieved the Goal
1. Installed **PyTorch** with CUDA support to ensure the GPU could be used for fast image generation.
2. Loaded the **Stable Diffusion** model using the `diffusers` library.
3. Implemented a loop to generate multiple images from a single text prompt.
4. Automatically saved each generated image with a numbered file name for easy identification.

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
3. Run the image generation script
4. Enter your prompt and specify how many images you'd like to generate. The images will be saved in the current directory with numbered file names.

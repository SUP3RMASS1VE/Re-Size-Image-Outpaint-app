# Re-Size Image Outpainting

A powerful tool for extending images to different aspect ratios using Stable Diffusion XL.

## Features

- Resize images to popular aspect ratios (9:16, 16:9, 1:1) or custom dimensions
- Control image placement with multiple alignment options (Middle, Left, Right, Top, Bottom)
- Adjust overlap percentage and mask settings
- Preview alignment and mask before generation
- Interactive UI with image history
- Use generated images as new inputs for further processing

## Requirements

- Python 3.8+
- CUDA-compatible GPU
- PyTorch with CUDA support

## Installation

1. Clone this repository
2. Install the required packages:
   ```
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
   pip install -r requirements.txt
   ```

## Usage

1. Run the application:
   ```
   python app.py
   ```
2. Open your browser at `http://127.0.0.1:7860`
3. Upload an image
4. Select your desired output aspect ratio and alignment
5. Adjust advanced settings if needed
6. Click "Generate" to start the outpainting process

## Advanced Settings

- **Target Width/Height**: Set specific dimensions for the output image
- **Steps**: Control the number of inference steps (higher = better quality but slower)
- **Mask Overlap**: Adjust how much of the original image is used for context
- **Resize Input Image**: Scale the input image before processing
- **Prompt**: Add optional text prompts to guide the generation

## Examples

The `examples` directory contains sample images to test the application.

## Credits

This project is based on the original work by [VIDraft on Hugging Face](https://huggingface.co/spaces/VIDraft/ReSize-Image-Outpainting). Thanks to the original creators for their contribution.

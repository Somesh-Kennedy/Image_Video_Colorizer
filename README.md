# Image and Video Colorizer

This project uses deep learning models to colorize grayscale images and videos automatically. The system applies neural networks to generate realistic colorizations from black-and-white input data.

## Features

* Colorizes static grayscale images.
* Colorizes grayscale video frames.
* Real-time processing with high-quality output.
* Implements state-of-the-art deep learning models for image and video colorization.

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/Somesh-Kennedy/Image_Video_Colorizer.git
   cd Image_Video_Colorizer
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

## Usage

### For Image Colorization:

```bash
python colorize_image.py --input path_to_grayscale_image --output path_to_output_image
```

### For Video Colorization:

```bash
python colorize_video.py --input path_to_grayscale_video --output path_to_output_video
```

## Requirements

* Python 3.x
* TensorFlow or PyTorch (depending on implementation)
* OpenCV

## Contributing

Feel free to fork this repository, create a branch, and submit a pull request with improvements or bug fixes.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.


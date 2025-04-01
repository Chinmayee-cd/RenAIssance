# Renaissance Text Generator

A deep learning system for generating synthetic Renaissance-style printed text with realistic printing imperfections.

## Project Overview

This project implements a generative model system that creates synthetic Renaissance-style printed text, incorporating realistic printing imperfections such as ink bleed, smudging, and faded text. The system uses a combination of diffusion models and GANs to achieve high-quality results.

## Features

- Text generation using diffusion models
- Realistic printing imperfections simulation
- Historical text style preservation
- Quantitative evaluation metrics
- Support for Spanish Renaissance text

## Project Structure

```
renaissance_text_generator/
├── data/                   # Dataset and processed data
├── models/                 # Model definitions and weights
├── utils/                  # Utility functions
├── evaluation/            # Evaluation metrics and tools
├── config/                # Configuration files
└── main.py               # Main execution script
```

## Setup

1. Create a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Download the dataset from the provided SharePoint links:

- Text scans: [Link to scans]
- Transcriptions: [Link to transcriptions]

## Usage

1. Prepare the dataset:

```bash
python utils/prepare_dataset.py
```

2. Train the model:

```bash
python main.py --mode train
```

3. Generate synthetic text:

```bash
python main.py --mode generate
```

4. Evaluate results:

```bash
python main.py --mode evaluate
```

## Evaluation Metrics

The system uses the following metrics to evaluate generated text:

1. **Structural Similarity Index (SSIM)**: Measures structural similarity between generated and real text
2. **Fréchet Inception Distance (FID)**: Evaluates the quality and diversity of generated images
3. **OCR Accuracy**: Measures how well the generated text can be recognized
4. **Historical Style Consistency**: Evaluates how well the generated text matches Renaissance style

## Results

The model generates synthetic Renaissance-style text with the following characteristics:

- Realistic printing imperfections
- Historical font styles
- Appropriate ink bleed and smudging effects
- Consistent text layout and formatting

## License

This project is licensed under the MIT License - see the LICENSE file for details.

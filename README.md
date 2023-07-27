# Sentiment Analysis with PyTorch

This project aims to perform sentiment analysis on textual content using PyTorch.

## Overview

- **Data Preprocessing**: Using Python libraries to prepare the data.
- **Feature Extraction**: Applying NLP techniques to extract information from text.
- **Model Development**: Creating, training, and evaluating models using PyTorch.

## Setup

### Requirements

- Python (version 3.7 or newer)
- PyTorch (version 1.7 or newer)
- Essential Libraries: pandas, NLTK, torchtext, sklearn

To install:
```bash
pip install -r requirements.txt
```

### Dataset

The project uses the IMDb Movie Reviews Dataset. Place it in the `data/` directory at the project's root.

### How to Use

1. Transform Data:
```bash
python preprocess.py
```

2. Train the Model:
```bash
python train.py
```

3. Evaluate the Model:
```bash
python evaluate.py
```

## Contributing

Fork, make your changes, and submit a pull request. For larger changes, start with an issue.

## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Acknowledgments

Thanks to IMDb for the movie reviews dataset and the PyTorch community for their resources.

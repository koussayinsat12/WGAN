# WGAN for Fashion MNIST

This project implements a Wasserstein Generative Adversarial Network (WGAN) for generating images of Fashion MNIST dataset using TensorFlow and Keras.

## Project Structure

The project consists of the following files:

- **model.py**: Contains the WGAN model definition.
- **config.py**: Configuration file with hyperparameters and model settings.
- **utils.py**: Utility functions, including model building and normalization.
- **train.py**: Script for training the WGAN on the Fashion MNIST dataset.
- **README.md**: Project documentation.

## Usage



1. **Training:**
    - Update hyperparameters and model configurations in `config.py` if necessary.
    - Run the training script:
        ```bash
        python train.py
        ```

2. **Generated Images:**
    - After training, generated images will be saved in the project directory.

## Files and Descriptions

- **model.py**: Defines the WGAN model class, including the generator, discriminator, and training methods.

- **config.py**: Configurations such as image shape, noise dimension, epochs, batch size, and optimizer settings.

- **utils.py**: Utility functions for building generator and discriminator models, normalization, and loss functions.

- **train.py**: Script for loading the Fashion MNIST dataset, initializing the WGAN model, and training.

## Results

Generated images will be saved in the project directory. You can visualize the results and monitor the training progress.

## Dependencies

- TensorFlow
- Keras
- TensorFlow Datasets

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

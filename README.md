# Practical Labs 8
# Implement CNN Model for Pet Image Classification

In this task, you will implement a Convolutional Neural Network (CNN) to classify images of cats and dogs. The dataset is a reduced version of the [PetsImages dataset](https://www.kaggle.com/c/dogs-vs-cats), containing 700 images for each class.

> Note: Due to the limited dataset size and simplicity of the CNN, the expected accuracy should be around 70-75%.

## Preparation

Before completing the task, you need to set up the development environment.

### Creating a Python Virtual Environment

1. For Windows:
   ```powershell
   python -m venv venv
   ```

2. For Linux/macOS:
   ```bash
   python3 -m venv venv
   ```

### Activating the Virtual Environment

1. For Windows:
   ```powershell
   .\venv\Scripts\activate
   ```

2. For Linux/macOS:
   ```bash
   source venv/bin/activate
   ```

> After activating the virtual environment, you will see the prefix `(venv)` before the command line.

### Installing Required Packages from `requirements.txt`

1. For both systems:
   ```bash
   pip install -r requirements.txt
   ```

## Tasks

### Prerequisites
1. Set up a Python virtual environment as described in the Preparation section above
2. Install required packages from `requirements.txt`
3. Download the dataset:
   - The dataset will be downloaded from Google Cloud Storage
   - Run `python download_dataset.py` in your terminal
   - The script will:
     - Check if the dataset already exists
     - Download a zip file containing the pet images
     - Extract the images to the `PetImages` directory
     - Create the following structure:
       ```
       PetImages/
       ├── train/
       │   ├── Cat/
       │   └── Dog/
       └── val/
           ├── Cat/
           └── Dog/
       ```
   - Each class (Cat/Dog) contains 700 images
   - The dataset is automatically split into training (90%) and validation (10%) sets

### Implementation Requirements

1. Implement the `CNN()` function that returns a Sequential model with the following architecture:
   - Input layer accepting images of size `180x180x3`
   - Three convolutional blocks, each containing:
     - Conv2D layer with increasing number of filters `(128, 256, 512)`
       - All Conv2D layers use kernel size `(3,3)`, stride `(2,2)`, `padding="same"`, and **ReLU** activation
     - BatchNormalization layer
     - MaxPooling2D layer with pool size `(3,3)`, stride `(2,2)`, and default padding
   - Flatten layer to convert 3D features to 1D
   - Dense layer with `1024` units and ReLU activation
   - Dropout layer with rate `0.3`
   - Output Dense layer with `2` units and **softmax** activation

2. Implement the `train()` function that:
   - Loads the dataset using the provided `train_dataset()` function with the specified batch size
   - Creates the CNN model using your implemented `CNN()` function
   - Compiles the model with:
     - Optimizer: 'adam'
     - Loss: 'categorical_crossentropy'
     - Metrics: ['accuracy']
   - Prints the model summary using `model.summary()`
   - Trains the model using `model.fit()` with:
     - Training data
     - Validation data
     - Number of epochs
     - `IntervalLogger` callback for progress tracking
     - `verbose=0` to use the custom logger
   - Evaluates the model on the test set using `model.evaluate()`
   - Prints the test accuracy
   - Returns a tuple containing:
     - The trained model
     - The test data generator

#### Important Notes
- Use appropriate activation functions for each layer
- Implement proper padding in convolutional layers
- Use the provided `IntervalLogger` callback for training progress
- The model should be compiled with 'adam' optimizer and 'categorical_crossentropy' loss
- Include accuracy as a metric during training
- Print the model summary before training
- Print the final test accuracy after training

#### Expected Output
When running the training, you should see:
1. Model architecture summary
2. Training progress at specified intervals
3. Final test accuracy (should be around 70-75%)

Remember that the dataset is intentionally small to make the task manageable, so don't expect very high accuracy. Focus on implementing the architecture correctly and understanding how each layer contributes to the model's performance.

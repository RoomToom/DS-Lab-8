import json
from pathlib import Path
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout, Input, BatchNormalization
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Optional, Dict, Any
from tensorflow.keras.preprocessing.image import DirectoryIterator

class IntervalLogger(tf.keras.callbacks.ProgbarLogger):
    """
    Custom callback for logging training progress at specified intervals.
    This helps reduce console output by only showing progress at certain epoch intervals.
    """
    def __init__(self, verbose: int = 1, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.default_verbose = verbose
        self.epoch_interval = None

    def on_train_begin(self, logs: Optional[Dict[str, float]] = None) -> None:
        super().on_train_begin(logs)
        total_epochs = self.epochs
        if total_epochs is None:
            self.epoch_interval = 10
        else:
            self.epoch_interval = max(1, total_epochs // 10)

    def on_epoch_begin(self, epoch: int, *args: Any, **kwargs: Any) -> None:
        if epoch == 0 or (epoch + 1) % self.epoch_interval == 0:
            self.verbose = self.default_verbose
        else:
            self.verbose = 0
        super().on_epoch_begin(epoch, *args, **kwargs)


def train_dataset(batch_size: int = 64) -> Tuple[DirectoryIterator, DirectoryIterator, DirectoryIterator]:
    """
    Prepare and load the training, validation, and test datasets with data augmentation.
    
    Args:
        batch_size (int): Number of samples per batch
        
    Returns:
        tuple: (train_generator, validation_generator, test_generator)
    """
    root = Path(__file__).resolve().parents[1]
    dir_train = root / 'PetImages/train'
    dir_test = root / 'PetImages/val'

    # Data augmentation configuration for training data
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        zoom_range=0.3,
        rotation_range=50,
        width_shift_range=0.1,
        height_shift_range=0.2,
        horizontal_flip=True,
        vertical_flip=True,
        validation_split=0.1,
    )
    # Only rescale test data, no augmentation
    val_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.1)
    test_datagen = ImageDataGenerator(rescale=1./255)

    # Load training data with augmentation
    train = train_datagen.flow_from_directory(
        dir_train,
        target_size=(180, 180),
        batch_size=batch_size,
        subset="training",
        class_mode='categorical',
        classes=['Cat', 'Dog'],
        seed=42
    )
    # Load validation data
    val = val_datagen.flow_from_directory(
        dir_train,
        target_size=(180, 180),
        batch_size=batch_size,
        subset="validation",
        shuffle=False,
        class_mode='categorical',
        classes=['Cat', 'Dog'],
        seed=42
    )
    # Load test data
    test = test_datagen.flow_from_directory(
        dir_test,
        shuffle=False,
        target_size=(180, 180),
        batch_size=batch_size,
        class_mode='categorical',
        classes=['Cat', 'Dog'],
        seed=42
    )

    return train, val, test


def CNN() -> Sequential:
    """
    Create and return a CNN model for pet image classification.
    
    Returns:
        Sequential: A CNN model with the following architecture:
        - Three convolutional blocks with increasing filters
        - Batch normalization and max pooling after each block
        - Dense layers with dropout for classification
    """
    model = Sequential([Input(shape=(180, 180, 3))])
    for filters in (128, 256, 512):
        model.add(Conv2D(filters, (3, 3), strides=(2, 2), padding="same", activation="relu"))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
    model.add(Flatten())
    model.add(Dense(1024, activation="relu"))
    model.add(Dropout(0.3))
    model.add(Dense(2, activation="softmax"))

    return model


def train(batch_size: int = 32, epochs: int = 10) -> Tuple[Sequential, DirectoryIterator]:
    """
    Train the CNN model on the pet images dataset.
    
    Args:
        batch_size (int): Number of samples per batch
        epochs (int): Number of training epochs
        
    Returns:
        tuple: (trained_model, test_data)
    """
    # Load the data
    keras.utils.set_random_seed(42)
    train_data, val_data, test_data = train_dataset(batch_size)
    
    # Create and compile the model
    model = CNN()
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

    # Print model summary
    model.summary()
    
    output = Path(__file__).resolve().parents[1] / 'results'
    output.mkdir(exist_ok=True)
    checkpoint = output.parent / 'pets_model.keras'
    # Select weights using validation data, leaving the test set for final evaluation.
    history = model.fit(train_data, validation_data=val_data, epochs=epochs,
                        callbacks=[IntervalLogger(), keras.callbacks.ModelCheckpoint(
                            checkpoint, monitor='val_loss', save_best_only=True)], verbose=0)
    model = keras.models.load_model(checkpoint)
    
    # Evaluate on test set
    test_data.reset()
    test_loss, test_acc = model.evaluate(test_data, verbose=0)
    print(f"Test accuracy: {test_acc:.4f}")
    
    pd.DataFrame(history.history, index=range(1, epochs + 1)).to_csv(
        output / 'history.csv', index_label='epoch')
    metrics = {'test_loss': float(test_loss), 'test_accuracy': float(test_acc),
               'epochs': epochs, 'best_epoch': int(np.argmin(history.history['val_loss']) + 1),
               'batch_size': batch_size,
               'training_images': train_data.samples, 'validation_images': val_data.samples,
               'test_images': test_data.samples, 'class_indices': test_data.class_indices}
    (output / 'metrics.json').write_text(json.dumps(metrics, indent=2) + '\n', encoding='utf-8')
    test_data.reset()
    predictions = model.predict(test_data, verbose=0)
    pd.DataFrame({'file': test_data.filenames, 'actual': test_data.classes,
                  'predicted': predictions.argmax(axis=1),
                  'probability_cat': predictions[:, 0], 'probability_dog': predictions[:, 1]}).to_csv(
        output / 'predictions.csv', index=False)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, metric in zip(axes, ['loss', 'accuracy']):
        ax.plot(range(1, epochs + 1), history.history[metric], label='Training')
        ax.plot(range(1, epochs + 1), history.history['val_' + metric], label='Validation')
        ax.set(xlabel='Epoch', ylabel=metric.capitalize())
        ax.legend()
    fig.tight_layout()
    fig.savefig(output / 'training.png', dpi=150)

    return model, test_data


def visualize_augmented_images(train_data: DirectoryIterator) -> None:
    """
    Visualize a batch of augmented training images with their labels.
    
    Args:
        train_data: Training data generator
    """
    # Get a batch of augmented images
    images, labels = next(train_data)
    class_names = {0: 'Cat', 1: 'Dog'}
    
    # Plot the images
    plt.figure(figsize=(10, 10))
    for i in range(min(16, len(images))):
        plt.subplot(4, 4, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(images[i])
        plt.xlabel(class_names[np.argmax(labels[i])])
    plt.tight_layout()


def visualize_predictions(model: Sequential, test_data: DirectoryIterator) -> None:
    """
    Visualize model predictions on test images.
    
    Args:
        model: Trained CNN model
        test_data: Test data generator
    """
    indices = np.concatenate([np.flatnonzero(test_data.classes == label)[:8]
                              for label in (0, 1)])
    images = np.stack([keras.utils.img_to_array(keras.utils.load_img(
        test_data.filepaths[i], target_size=(180, 180))) / 255. for i in indices])
    true_labels = keras.utils.to_categorical(test_data.classes[indices], num_classes=2)
    predictions = model.predict(images, verbose=0)
    class_names = {0: 'Cat', 1: 'Dog'}
    
    # Plot the predictions
    plt.figure(figsize=(15, 15))
    for i in range(min(16, len(images))):
        plt.subplot(4, 4, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(images[i])
        predicted_class = class_names[np.argmax(predictions[i])]
        true_class = class_names[np.argmax(true_labels[i])]
        plt.xlabel(f"Pred: {predicted_class}\nTrue: {true_class}")
    plt.tight_layout()
    plt.savefig(Path(__file__).resolve().parents[1] / "results/predictions.png", dpi=150)
    plt.show()


if __name__ == '__main__':
    output = Path(__file__).resolve().parents[1] / 'results'
    output.mkdir(exist_ok=True)
    keras.utils.set_random_seed(42)
    train_data, val_data, test_data = train_dataset()
    
    # Visualize augmented images
    visualize_augmented_images(train_data)
    plt.savefig(output / "augmented_images.png", dpi=150)
    
    # Train the model
    model, test_data = train(epochs=30)
    
    # Visualize predictions
    visualize_predictions(model, test_data)

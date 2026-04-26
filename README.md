# Trash Classification with Transfer Learning using VGG16

Automatically classify waste products using image recognition techniques.

This project aims to build a model that can differentiate between recyclable and organic waste products using transfer learning.

## Steps done in this project

1. Import required libraries and check package installation.
2. Download and extract the waste image dataset.
3. Configure image size, batch size, classes, and directory paths.
4. Create image generators for training, validation, and testing.
5. Visualize some augmented images.
6. Load VGG16 as the pre-trained base model.
7. Flatten the VGG16 output and create a new classifier head on top.
8. Compile and train the extract-features model.
9. Plot loss and accuracy curves.
10. Fine-tune the model by unfreezing part of the base network.
11. Evaluate both models on test data.
12. Check model predictions on sample test images.

## Dataset and labels

I train an algorithm on images and to predict the labels for images in my test set (`1 = recyclable`, `0 = organic`).

Folder structure used in the notebook:

- `o-vs-r-split/train/O`
- `o-vs-r-split/train/R`
- `o-vs-r-split/test/O`
- `o-vs-r-split/test/R`

## Loading images using ImageDataGenerator

I create `ImageDataGenerators` used for training, validation and testing. Image data generators create batches of tensor image data with real-time data augmentation.

Main preprocessing used:

- rescaling by `1/255`
- width and height shifts
- horizontal flip
- validation split of `0.2`

Configuration used:

- image size: `150 x 150`
- batch size: `32`
- epochs: `10`
- classes: `2`

## Load VGG16

The project uses transfer learning with VGG16:

- `VGG16(include_top=False, weights='imagenet', input_shape=(150, 150, 3))`

Then the output of the VGG model is flattened and grouped into a base model for training and inference.

## Create a new model on top

The classifier added on top of VGG16 is:

- base VGG16 feature extractor
- `Flatten`
- `Dense(512, activation='relu')`
- `Dropout(0.3)`
- `Dense(512, activation='relu')`
- `Dropout(0.3)`
- `Dense(1, activation='sigmoid')`

This is a binary classifier:
- sigmoid output close to `1` means recyclable
- sigmoid output close to `0` means organic

## Compile the model using `model.compile()`

The model is compiled with:

- loss: `binary_crossentropy`
- optimizer: `Adam(learning_rate=1e-5)`
- metric: `accuracy`

### Why these choices were used

- `binary_crossentropy` is correct because this is a binary classification task
- `sigmoid` is used in the last layer for binary probability output
- `Adam` is a strong default optimizer and works well for transfer learning
- the small learning rate helps avoid damaging the pre-trained VGG16 features too quickly

## I use early stopping to avoid over-training the model

I define training callbacks. Callbacks are extra instructions that run during training.

The notebook uses:

- `EarlyStopping`
- `ModelCheckpoint`
- `LearningRateScheduler`

These help stop unnecessary training, save the best model, and reduce the learning rate over epochs.

## Fit and train the model

The first model is trained with the VGG16 base frozen, so it works as an extract-features model.

From the notebook results, the first model reached about:

- validation accuracy around `0.81`
- test accuracy `0.81`

## Fine-tuning the model

An optional step in transfer learning, it usually ends up improving the performance of the model.

The notebook unfreezes part of the base model starting from:

- `block5_conv3`

So the top of VGG16 is allowed to adapt to the trash dataset, while most earlier layers remain frozen.

The fine-tuned model improved the result slightly:

- test accuracy `0.83`

## Evaluation of both model on test data

### Extract Features Model

- accuracy: `0.81`
- class `O` f1-score: `0.83`
- class `R` f1-score: `0.79`

### Fine-Tuned Model

- accuracy: `0.83`
- class `O` f1-score: `0.84`
- class `R` f1-score: `0.81`

## Result analysis

The fine-tuned model performs slightly better than the extract-features model.

Main observations:

- fine-tuning improved overall accuracy from `0.81` to `0.83`
- class `O` is detected a bit more reliably than class `R`
- recall for recyclable waste (`R`) is lower than for organic waste (`O)`, so the model misses some recyclable items
- the improvement is real but modest, which is expected because only a small part of VGG16 was unfrozen

So the project shows that transfer learning works reasonably well for this binary trash-classification task, and fine-tuning gives a small but useful gain.

## Check model

The notebook also plots individual test images with:

- model name
- actual label
- predicted label

This is useful for manually inspecting which examples are easy or difficult for the model.

## Packages used

- `tensorflow`
- `numpy`
- `pandas`
- `scikit-learn`
- `matplotlib`
- `seaborn`
- `pyarrow`
- `requests`
- `glob`
- `os`

Main Keras/TensorFlow components used:

- `ImageDataGenerator`
- `VGG16`
- `Sequential`
- `Model`
- `Dense`
- `Dropout`
- `Flatten`
- `EarlyStopping`
- `ModelCheckpoint`
- `LearningRateScheduler`

## Files

- `Keras-TrashClassification-TransferLearning(1).ipynb`
- `README.md`
- `O_R_tlearn_vgg16.keras`
- `O_R_tlearn_fine_tune_vgg16.keras`

## Summary

This project uses transfer learning with VGG16 to classify waste images into recyclable and organic categories. It first trains a classifier on top of frozen VGG16 features, then fine-tunes the top of the base model for better performance. The extract-features model reaches about `81%` accuracy, and the fine-tuned model improves that to about `83%`, showing that transfer learning is effective for this binary image-classification problem.

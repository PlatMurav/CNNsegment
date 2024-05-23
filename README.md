# Segmentation Model with TensorFlow and TFRecords
## Overview
This repository contains code for building a semantic segmentation model using TensorFlow and TFRecords.
<br>The model architecture used in this repository is a Convolutional Neural Network (CNN) based on U-Net and designed for semantic segmentation tasks. It takes input images and outputs segmentation masks with class labels for each pixel.

### Dataset
The datasets used for training, validation and testing the model consist of images and their corresponding segmentation masks.
<br>The images are stored in a folder, and the annotations are provided in COCO format (JSON-file).
<br>The dataset is preprocessed and converted into TFRecord format for efficient storage and processing.

### TFRecords
TFRecords are a binary format for storing data in TensorFlow.
In this project, TFRecords are used to store preprocessed images and masks with their parameters, allowing for efficient data reading during training.

## Requirements
To be albe to use the model yuo should have:
* Images
* JSON-file in coco-format

A folder with images that may contain some of the 10 objects and JSON-file (in coco-format) with annotations.

All necessary **dependencies** You can install using pip:
```python
pip install -r requirements.txt
```

## Usage
*based on main.py*
### Creating a new model
If you want to train a new model you simply need to copy steps that are shown in main.py.
<br>At first, you need:
1. import our class
2. create an instance
3. pass all necessary argumets (path to the folder with images and path to json-file)
```python
from model import Segmentator

model = Segmentator('images', 'instances.json')
```
Now pre-model is created and we can start using it. 

### Splitting the data and creating TfRecodrs
To operate images and their masks and store them as TfRecords we need to split them at first.
<br>We can do this by using .split_data() metohd:
```python
model.split_data()
```
It splits images, their id and saves them internally. After splitting the data we should just call **.create_tfrecords()** method and pass arguments (ouput path and ['train', 'val', 'test']):
```python
# Creating output paths
train_tfrecord_file = 'train.tfrecord'   # or 'tfrecords/train.tfrecord'
val_tfrecord_file = 'val.tfrecord'
test_tfrecord_file = 'test.tfrecord'

# Writing TfRecords
model.create_tfrecords(train_tfrecord_file, "train")
model.create_tfrecords(val_tfrecord_file, 'val')
model.create_tfrecords(test_tfrecord_file, 'test')
```
After this step we'll see three (if we called it three times) files in the current directory: *train.tfrecord*, *val.tfrecord*, *test.tfrecord*.
<br>This files contain images and their masks in binary format

### Creating Datasets
When we have tfrecord files we need to create datasets so we could start training.
<br>All we have to do is just call **.create_dataset()** method and save them in some variables:
```python
# converts TfRecords into datasets
train_dataset = model.create_dataset(train_tfrecord_file)
val_dataset = model.create_dataset(val_tfrecord_file)
test_dataset = model.create_dataset(test_tfrecord_file)
```

### Initializing model and EarlyStopping
(**optional**): If we want to add Early Stoping we should call the method :
```python
model.define_early_stopping(patience=7)
```
Now we can Initialize our model by writing:
```python
model.build_model()
```
This method creates CNN-structure with 6 convolutional layers internally and sets Adam optimizer.

### Training and evalation
To start trainig the model we call **.build_model()** method and pass training and validation datasets which were created earlier:
```python
model.train_model(train_dataset, val_dataset)
```
**Notes**:
* Internally it uses **parse_function()** function to parse TFRecord examples during training. It decodes the serialized examples and prepares them as input for the model.
* The model is trained using the TensorFlow framework: loads the dataset from TFRecord files, preprocesses the data and trains the CNN model using optimization.

If we want to evaluate our model and we are sure it's ready for the final test (all hyperparameters are set, and the whole model is tuned):
```python
test_loss, test_accuracy = segment.eval_set(test_dataset)
print('Test loss:', test_loss)
print('Test accuracy:', test_accuracy)
```
We call **.eval_set()** method providing test dataset for evaluation.

### Saving and loading the model
When our model is ready we can save it providing path to the method:
```python
#native Keras format
model.save_model('trained_model_t800.keras')
```
If we already a file with trained model we can load it using:
```python
model.load_model('trained_model_t800.keras')
```


import json
import numpy as np
import tensorflow as tf
import os
from pycocotools.coco import COCO
import cv2
from sklearn.model_selection import train_test_split


class Segmentator:

    def __init__(self, images_path, coco_path):
        self.images_path = images_path
        self.model = None
        self.batch_size = 28          # default value
        self.epochs = 55
        self.val_images_info = None
        self.test_images_info = None
        self.train_images_info = None
        self.history = None
        self.coco = COCO(coco_path)
        self.callbacks = []

    def define_early_stopping(self, patience=7):
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',       # Metric to monitor
                                                          patience=patience,    # Number of epochs
                                                          verbose=1,     # information about early_stopping
                                                          restore_best_weights=True  # Restore model weights
                                                          )
        self.callbacks.append(early_stopping)

    def set_epochs(self,
                   epochs: int):
        """
        Sets the number of epochs to be used during model training the model. It checks whether the input
        value is a positive integer and assigns it to the `epochs` attribute of the class instance.
        """
        if isinstance(epochs, int) and epochs > 0:
            self.epochs = epochs

    def set_batch_size(self,
                       batch_size: int):
        """
        Sets the batch size for training and evaluation of the model.

        Args:
            batch_size (int): The batch size for training and evaluation.
        """
        if isinstance(batch_size, int) and batch_size > 0:
            self.batch_size = batch_size

    def create_tfrecords(self,
                         output_file: str,
                         image_info_list: str):
        """
        Creates a TFRecord file from a list with annotations, created from JSON file in COCO-format.

        This function reads images and their corresponding masks from the provided directory and JSON (coco) file respectavily,
        encodes them into the TFRecord format, and writes the resulting records to the specified output file.
        Each record in the TFRecord file will contain height and width along with its mask.

        Args:
            output_file (str): Path where the output TFRecord file will be saved or just its name.
            image_info_list (lst): list containing annotations in COCO format.
        """
        with tf.io.TFRecordWriter(output_file) as writer:
            if image_info_list == 'train':
                image_info_list = self.train_images_info
            elif image_info_list == 'val':
                image_info_list = self.val_images_info
            elif image_info_list == 'test':
                image_info_list = self.test_images_info
            else:
                print("Can't process the data, the third parameter must be in ['train', 'test', 'val']")
                return
            for image_info in image_info_list:
                image_id = image_info['id']
                file_name = image_info['file_name']
                full_image_path = os.path.join(self.images_path, file_name)

                image = cv2.imread(full_image_path)

                resized_img = cv2.resize(image, (240, 320))
                height, width, _ = resized_img.shape
                image_string = resized_img.tobytes()  # string of bytes

                mask = self.__create_masks(image_id, self.coco)
                resized_mask = cv2.resize(mask, (240, 320))
                tf_example = self.__image_example(image_string, resized_mask, height, width)
                writer.write(tf_example.SerializeToString())

    def __create_masks(self,
                       image_id: int,
                       coco: COCO):
        """
        Creates the mask of an image
            Args:
                image_id (int): The input image id
                coco (pycocotools.coco.COCO): json-file in coco format

            Returns:
                mask (np.array: (1024, 768)): mask of an image
        """
        ann_ids = coco.getAnnIds(imgIds=image_id)
        anns = coco.loadAnns(ann_ids)
        mask = np.zeros((1024, 768), dtype=np.uint8)

        for ann in anns:
            category_id = ann['category_id']
            binary_mask = coco.annToMask(ann)
            mask[binary_mask == 1] = category_id
        return mask

    def create_dataset(self, tfrecord_file):
        """
        Creates a TensorFlow Dataset from a TFRecord file.

        This function reads a TFRecord file and parses each example into a TensorFlow Dataset object.
        The dataset will contain image and mask pairs along with height and width, where images and masks are decoded and resized.

        Args:
            tfrecord_file (str): Path to the TFRecord file.

        Returns:
            tf.data.Dataset: A TensorFlow Dataset object containing parsed image and mask pairs.

        The function performs the following steps:
            1. Reads the TFRecord file using `tf.data.TFRecordDataset`.
            2. Parses each record using the `_parse_function` to extract and process the image and its mask.
            3. Returns the resulting dataset for further training.
        """
        dataset = tf.data.TFRecordDataset(tfrecord_file)
        dataset = dataset.map(self.__parse_function)
        dataset = dataset.shuffle(buffer_size=100)
        dataset = dataset.batch(self.batch_size)
        dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)  # remove or leave?
        return dataset

    def __image_example(self,
                        image_string: bytes,
                        mask,
                        height: int,
                        width: int):
        """
        Creates a TensorFlow Example from an image and its corresponding mask.

        This function takes an encoded image string, its mask, and the dimensions of the image,
        and constructs a TensorFlow Example protocol buffer. The resulting Example contains the
        height and width of the image, the raw bytes of the image, and the raw bytes of the mask.

        Args:
            image_string (bytes): The raw bytes of the encoded image.
            mask (np.ndarray): The mask corresponding to the image, represented as a NumPy array.
            height (int): The height of the image.
            width (int): The width of the image.

        Returns:
            tf.train.Example: A TensorFlow Example protocol buffer containing the image and mask data.

        This Example can be serialized and written to a TFRecord file for efficient storage and processing.
        """
        feature = {
            'height': tf.train.Feature(int64_list=tf.train.Int64List(value=[height])),
            'width': tf.train.Feature(int64_list=tf.train.Int64List(value=[width])),
            'image_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_string])),
            'mask_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[mask.tobytes()])),
        }
        return tf.train.Example(features=tf.train.Features(feature=feature))

    def __parse_function(self, proto) -> tuple:
        """
        Parses a single example from a TFRecord file into an image and its mask.

        This function is used to parse the input TFRecord file which contains encoded images and their respective masks.
        It decodes the image, resizes it to the desired dimensions, and normalizes it to the [0, 1] range.

        Args:
            proto (tf.Tensor): A scalar Tensor of type string containing a serialized Example protocol buffer.

        Returns:
            tuple: A tuple of two elements:
                - image (tf.Tensor): A Tensor of shape (320, 240, 3) representing the resized and normalized image.
                - mask (tf.Tensor): A Tensor of type uint8 representing the mask of the image.

        The example protocol buffer expected in the TFRecord file should have the following features:
            - 'height': An integer representing the height of the reshaped image.
            - 'width': An integer representing the width of the reshaped image.
            - 'image_raw': A string containing the raw bytes of the encoded image.
            - 'mask_raw': A string containing the raw bytes of the encoded mask of an image.
        """
        keys_to_features = {
            'height': tf.io.FixedLenFeature([], tf.int64),
            'width': tf.io.FixedLenFeature([], tf.int64),
            'image_raw': tf.io.FixedLenFeature([], tf.string),
            'mask_raw': tf.io.FixedLenFeature([], tf.string),
        }
        parsed_features = tf.io.parse_single_example(proto, keys_to_features)

        height = tf.cast(parsed_features['height'], tf.int32)
        width = tf.cast(parsed_features['width'], tf.int32)

        image = tf.io.decode_raw(parsed_features['image_raw'], tf.int8)
        image = tf.reshape(image, [height, width, 3])
        image = tf.cast(image, tf.float32)
        image = image / 255.0

        mask = tf.io.decode_raw(parsed_features['mask_raw'], tf.uint8)
        mask = tf.reshape(mask, [height, width])

        return image, mask

    def split_data(self):
        image_ids = self.coco.getImgIds()
        images_info = self.coco.loadImgs(image_ids)
        train_ids, test_ids = train_test_split(image_ids, test_size=0.3, random_state=42)
        val_ids, test_ids = train_test_split(test_ids, test_size=0.5, random_state=42)
        self.train_images_info = [img_info for img_info in images_info if img_info['id'] in train_ids]
        self.val_images_info = [img_info for img_info in images_info if img_info['id'] in val_ids]
        self.test_images_info = [img_info for img_info in images_info if img_info['id'] in test_ids]

    def build_model(self, input_size=(320, 240, 3)):
        inputs = tf.keras.layers.Input(input_size)

        # Encoder
        conv1 = tf.keras.layers.Conv2D(16, 3, activation='relu', padding='same')(inputs)
        pool1 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv1)

        conv2 = tf.keras.layers.Conv2D(32, 3, activation='relu', padding='same')(pool1)
        pool2 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv2)

        # Bottleneck
        conv3 = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same')(pool2)

        # Decoder
        up4 = tf.keras.layers.Conv2D(32, 2, activation='relu', padding='same')(
            tf.keras.layers.UpSampling2D(size=(2, 2))(conv3))
        merge4 = tf.keras.layers.concatenate([conv2, up4], axis=3)
        conv4 = tf.keras.layers.Conv2D(32, 3, activation='relu', padding='same')(merge4)

        up5 = tf.keras.layers.Conv2D(16, 2, activation='relu', padding='same')(
            tf.keras.layers.UpSampling2D(size=(2, 2))(conv4))
        merge5 = tf.keras.layers.concatenate([conv1, up5], axis=3)
        conv5 = tf.keras.layers.Conv2D(16, 3, activation='relu', padding='same')(merge5)

        conv6 = tf.keras.layers.Conv2D(11, 1, activation='softmax')(conv5)  # 11 - the number of classes

        model = tf.keras.models.Model(inputs=inputs, outputs=conv6)

        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])

        self.model = model

    def save_model(self, path_to_save):
        """
        This function saves the current state of the model, including its architecture, weights, and configuration,
        to a specified file path.

        Args:
            path_to_save (str): The file path where the model should be saved (filepath or directory).
        """
        self.model.save(path_to_save)

    def load_model(self, path_to_load: str):
        """
        This function loads a pre-trained Keras model from a given file path. It first checks if the file exists
        at the specified path.

        Args:
            path_to_load (str): The file path from which the model should be loaded.
        """
        # Checks if the file or directory specified by `path_to_load` exists.
        if not os.path.exists(path_to_load):
            print("File does not exist")
            return  # or raise FileNotFoundError("File does not exist") if you want to raise an exception

        try:
            # If the file exists, attempts to load the model using `tf.keras.models.load_model`.
            self.model = tf.keras.models.load_model(path_to_load)
        except Exception as e:
            # If an error occurs during the loading process, catches the exception and prints an error message.
            print("An error occurred while loading the model:", e)

    def train_model(self, train_dataset, val_dataset):
        """
        Trains the CNN model on the provided training dataset and validates it on the validation dataset.
        It also supports early stopping through the use of callbacks to prevent overfitting.
g
        Args:
            train_dataset (tf.data.Dataset): The TensorFlow Dataset object containing the training data.

            val_dataset (tf.data.Dataset): The TensorFlow Dataset object containing the validation data.

        Notes:
            - `self.epochs` the number of epochs to train the model.
            - `self.callbacks` defined within the class and includes an EarlyStopping callback.
            - `self.model` a pre-built and compiled Keras Model object.
            - Training history is stored in `self.history` for further analysis.
        """
        self.history = self.model.fit(
            train_dataset,
            epochs=self.epochs,
            validation_data=val_dataset,
            callbacks=self.callbacks  # Include EarlyStopping callback
        )

    def eval_set(self, dataset):
        test_loss, test_accuracy = self.model.evaluate(dataset)
        return test_loss, test_accuracy







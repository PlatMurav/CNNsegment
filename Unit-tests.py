import unittest
from unittest.mock import patch, MagicMock
import os
import tensorflow as tf
import numpy as np
from pycocotools.coco import COCO
from model import Segmentator

class TestSegmentator(unittest.TestCase):

    @patch('model.COCO')
    def setUp(self, MockCOCO):
        # Setup a mocked COCO object
        self.mock_coco = MockCOCO()
        self.mock_coco.getImgIds.return_value = [1, 2, 3, 4, 5, 6]
        self.mock_coco.loadImgs.return_value = [{'id': 1, 'file_name': 'camera_1.png'},
                                                {'id': 2, 'file_name': 'camera_2.png'},
                                                {'id': 3, 'file_name': 'camera_3.png'},
                                                {'id': 4, 'file_name': 'camera_4.png'},
                                                {'id': 5, 'file_name': 'camera_5.png'},
                                                {'id': 6, 'file_name': 'camera_6.png'}]
        self.segmentator = Segmentator(images_path='images', coco_path='instances.json')

    def test_set_epochs(self):
        self.segmentator.set_epochs(40)
        self.assertEqual(self.segmentator.epochs, 40)
        self.segmentator.set_epochs(-1)
        self.assertNotEqual(self.segmentator.epochs, -1)

    def test_set_batch_size(self):
        self.segmentator.set_batch_size(32)
        self.assertEqual(self.segmentator.batch_size, 32)
        self.segmentator.set_batch_size(-1)
        self.assertNotEqual(self.segmentator.batch_size, -1)

    @patch('model.tf.io.TFRecordWriter')
    @patch('model.cv2.imread')
    @patch('model.cv2.resize')
    def test_create_tfrecords(self, mock_resize, mock_imread, MockTFRecordWriter):
        mock_imread.return_value = np.zeros((1024, 768, 3), dtype=np.uint8)
        mock_resize.return_value = np.zeros((320, 240, 3), dtype=np.uint8)
        mock_writer = MockTFRecordWriter.return_value.__enter__.return_value

        self.segmentator.train_images_info = [{'id': 1, 'file_name': 'camera_1.png'},
                                                {'id': 2, 'file_name': 'camera_2.png'}]
        self.segmentator._Segmentator__create_masks = MagicMock(return_value=np.zeros((1024, 768), dtype=np.uint8))
        self.segmentator._Segmentator__image_example = MagicMock(return_value=tf.train.Example())

        self.segmentator.create_tfrecords('output.tfrecord', 'train')
        self.assertEqual(mock_writer.write.call_count, 2)

    def test_split_data(self):
        self.segmentator.split_data()
        self.assertEqual(len(self.segmentator.train_images_info), 4)
        self.assertEqual(len(self.segmentator.val_images_info), 1)
        self.assertEqual(len(self.segmentator.test_images_info), 1)

    def test_build_model(self):
        self.segmentator.build_model()
        self.assertIsInstance(self.segmentator.model, tf.keras.Model)
        self.assertEqual(len(self.segmentator.model.layers), 15)

    @patch('model.tf.keras.models.Model')
    def test_save_model(self, MockModel):
        mock_model = MockModel.return_value
        self.segmentator.model = mock_model
        self.segmentator.save_model('fake_model_path')
        mock_model.save.assert_called_with('fake_model_path')


    @patch('model.tf.keras.models.load_model')
    @patch('model.os.path.exists', return_value=True)
    def test_load_model(self, mock_path_exists, mock_load_model):
        mock_load_model.return_value = tf.keras.Model()
        self.segmentator.load_model('saved_test_model.keras')
        mock_load_model.assert_called_once_with('saved_test_model.keras')

    @patch('model.tf.keras.models.load_model')
    @patch('model.os.path.exists', return_value=False)
    def test_load_model_file_not_exist(self, mock_path_exists, mock_load_model):
        self.segmentator.load_model('saved_test_model.keras')
        mock_load_model.assert_not_called()

    @patch('model.tf.data.TFRecordDataset')
    def test_create_dataset(self, MockTFRecordDataset):
        mock_dataset = MagicMock()
        mock_dataset.map.return_value = mock_dataset
        mock_dataset.shuffle.return_value = mock_dataset
        mock_dataset.batch.return_value = mock_dataset
        mock_dataset.prefetch.return_value = mock_dataset
        MockTFRecordDataset.return_value = mock_dataset

        dataset = self.segmentator.create_dataset('fake_file.tfrecord')
        self.assertEqual(dataset, mock_dataset)

    @patch('model.tf.keras.callbacks.EarlyStopping')
    def test_define_early_stopping(self, MockEarlyStopping):
        self.segmentator.define_early_stopping(patience=10)
        MockEarlyStopping.assert_called_once_with(monitor='val_loss', patience=10, verbose=1, restore_best_weights=True)
        self.assertEqual(len(self.segmentator.callbacks), 1)

    @patch('model.tf.keras.Model.fit')
    def test_train_model(self, mock_fit):
        self.segmentator.model = tf.keras.Model()
        train_dataset = MagicMock()
        val_dataset = MagicMock()
        self.segmentator.callbacks = [MagicMock()]
        self.segmentator.train_model(train_dataset, val_dataset)
        mock_fit.assert_called_once_with(train_dataset, epochs=self.segmentator.epochs, validation_data=val_dataset,
                                         callbacks=self.segmentator.callbacks)


if __name__ == '__main__':
    unittest.main()

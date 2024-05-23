from model import Segmentator

segment = Segmentator('images', 'instances.json')
# previously we created tfrecords, datasets and trained the model
segment.load_model('trained_model_t800.keras')

# since we already have test.tfrecord in the folder we don't need to call .create_tfrecords
test_tfrecord_file = 'test.tfrecord'
test_dataset = segment.create_dataset(test_tfrecord_file)

test_loss, test_accuracy = segment.eval_set(test_dataset)
print('Test loss:', test_loss)
print('Test accuracy:', test_accuracy)
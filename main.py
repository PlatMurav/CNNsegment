from model import Segmentator

# if __name__ == "__main__":

model = Segmentator('images', 'instances.json')
# we need to split our data we'll have test, train and validation sets
model.split_data()

train_tfrecord_file = 'tfrecords/train.tfrecord'
val_tfrecord_file = 'tfrecords/val.tfrecord'
test_tfrecord_file = 'tfrecords/test.tfrecord'

# Write TfRecords
model.create_tfrecords(train_tfrecord_file, "train")
model.create_tfrecords(val_tfrecord_file, 'val')
model.create_tfrecords(test_tfrecord_file, 'test')

# convert TfRecords into datasets so we could train the model
train_dataset = model.create_dataset(train_tfrecord_file)
val_dataset = model.create_dataset(val_tfrecord_file)
test_dataset = model.create_dataset(test_tfrecord_file)

# Initialize our model and EarlyStopping
model.define_early_stopping()
model.build_model()

# Training (built-in method fit)
model.train_model(train_dataset, val_dataset)

from random import shuffle
import tensorflow
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Flatten
import os
import wandb
from wandb.keras import WandbCallback

# ['blind-stick', 'normal-stick']
# ['chair', 'electric-wheelchair', 'wheelchair']
# ['long', 'square']
# ['chair', 'electric-wheelchair', 'stroller', 'wheelchair']

# gpu_options = tensorflow.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.2)
# sess = tensorflow.compat.v1.Session(config=tensorflow.compat.v1.ConfigProto(gpu_options=gpu_options))

# physical_devices = tensorflow.config.experimental.list_physical_devices('GPU')
# # assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
# config = tensorflow.config.experimental.set_memory_growth(physical_devices[0], True)

wandb.init(project="classification", entity="thainq")

SIZE = 300
NUM_CLASSES = 2

def define_model():
	custom_model = Sequential()
	model = tensorflow.keras.applications.EfficientNetB0(include_top=False, input_shape=(SIZE, SIZE, 3), classes=NUM_CLASSES,
													pooling='avg')
	for layer in model.layers:
		layer.trainable = False
	
	custom_model.add(model)
	custom_model.add(Flatten())
	custom_model.add(Dense(NUM_CLASSES, activation='softmax'))

	custom_model.compile(optimizer=tensorflow.keras.optimizers.Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
	return custom_model


def run():
	model = define_model()

	if not os.path.isdir('./20220818-2class-wheelchair-effb0'):
		os.mkdir('./20220818-2class-wheelchair-effb0')

	filepath = "{epoch:02d}-{val_accuracy:.4f}-20220818-2class-drone-effb0.h5"
	checkpoint = tensorflow.keras.callbacks.ModelCheckpoint(
		'./20220818-2class-wheelchair-effb0/' + filepath,
		monitor='val_accuracy',
		verbose=1,
		save_weight_only=False,
		mode='max')

	train_it = tensorflow.keras.preprocessing.image_dataset_from_directory(
		'/home/sonhh/thanhnt/classification_drone/waterfall/train/',
		image_size=(SIZE, SIZE),
		batch_size=4,
		seed=123,
		label_mode='categorical',
		color_mode="rgb",
		shuffle=True
	)

	val_it = tensorflow.keras.preprocessing.image_dataset_from_directory(
		'/home/sonhh/thanhnt/classification_drone/waterfall/val',
		image_size=(SIZE, SIZE),
		batch_size=4,
		seed=123,
		label_mode='categorical',
		color_mode="rgb",
		shuffle=True
	)
	print(train_it.class_names)
	history = model.fit(train_it, validation_data=val_it, epochs=50, verbose=1, callbacks=[checkpoint, WandbCallback()])


run()
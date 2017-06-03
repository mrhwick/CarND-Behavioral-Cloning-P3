import os
import csv
import cv2
import numpy as np
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

"""
Data loading script provided below.
"""

# Load the driving log samples provided.

samples = []
data_directory = 'my-training-data'
driving_log_path = data_directory + '/driving_log.csv'
with open(driving_log_path) as csvfile:
	print('reading provided driving log')
	reader = csv.reader(csvfile)
	for line in reader:
		samples.append(line)
print('received {} samples'.format(len(samples)))

# Split the training and validation sets out of those samples.

print('Splitting driving log samples into training and validation samples')
training_samples, validation_samples = train_test_split(samples, test_size=0.25)
print('{} training samples split'.format(len(training_samples)))
print('{} validation samples split'.format(len(validation_samples)))

# Define the generator function that can be used to get batches of samples

def generator(samples, batch_size=32):
	adjustment = 0.2
	num_samples = len(samples)
	# As long as the generator is iterated, provide more batches
	while True:
		# Shuffle the samples before splitting out the batches

		np.random.shuffle(samples)

		# Loop over the samples to generate batches.

		for offset in range(0, num_samples, batch_size):

			# Get this batch of samples from the overall listing.

			batch_samples = samples[offset:offset+batch_size]

			# Load the three images for this sample, and add the steering angles for each.

			images = []
			steering_angles = []
			for batch_sample in batch_samples:
				for i in range(3):
					# Load the image for this sample.

					name = data_directory + '/IMG/' + batch_sample[i].split('/')[-1]
					image = cv2.imread(name)
					image = image[...,::-1]

					# Load the steering angles for this sample.

					angle = float(batch_sample[3])

					# Add the image to the list for this batch.

					images.append(image)

					# If this is the left or right image, adjust the steering angle before adding.

					if i == 1:
						angle += adjustment
					if i == 2:
						angle -= adjustment

					# Add the steering angle to the list for this batch.

					steering_angles.append(angle)

			# Perform a flipping augmentation for each of the three images.

			augmented_images = []
			augmented_angles = []
			for image, angle in zip(images, steering_angles):
				# Add the originals to the new list.

				augmented_images.append(image)
				augmented_angles.append(angle)

				# Flip the image and the steering angle.

				flipped_image = image.copy()
				flipped_image = cv2.flip(flipped_image, 1)
				flipped_angle = angle * -1.0

				# Add the augmented versions to the new list.

				augmented_images.append(flipped_image)
				augmented_angles.append(flipped_angle)

			# Cast the augmented batch into numpy arrays.

			X_data = np.array(augmented_images)
			y_data = np.array(augmented_angles)

			# Yield the images shuffled, to obscure the augmentations.

			yield shuffle(X_data, y_data)

# Create the generators for our training and validation sets.

training_generator = generator(training_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)






"""
Keras Model provided below.
"""

from keras.models import Sequential, load_model
from keras.layers import Activation, Lambda, Flatten, Dense, MaxPooling2D, Dropout, Cropping2D
from keras.layers.convolutional import Conv2D
from keras.optimizers import Adam

load_model_for_retraining = True

if not load_model_for_retraining:
	# Create the model based off of the Nvidia model structure.

	model = Sequential()
	model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160, 320, 3)))
	model.add(Cropping2D(cropping=((70, 25), (0, 0))))
	model.add(Conv2D(24, 5, 5, subsample=(2, 2), activation='relu'))
	model.add(Conv2D(36, 5, 5, subsample=(2, 2), activation='relu'))
	model.add(Conv2D(48, 5, 5, subsample=(2, 2), activation='relu'))
	model.add(Conv2D(64, 3, 3, activation='relu'))
	model.add(Dropout(0.4))
	model.add(Conv2D(64, 3, 3, activation='relu'))
	model.add(Flatten())
	model.add(Dense(100))
	model.add(Dropout(0.2))
	model.add(Dense(50))
	model.add(Dropout(0.1))
	model.add(Dense(10))
	model.add(Dense(1))
else:
	# Load the model that was previously trained.
	model = load_model('model.h5')

# Create an ADAM optimizer.

optimizer = Adam()

# Use the ADAM optimizer and mean squared error for compilation.

model.compile(loss='mse', optimizer=optimizer)

# Train the model for 15 epochs. Takes about ~45 minutes on AWS GPU instances.

model.fit_generator(
	training_generator,
	samples_per_epoch=len(training_samples)*6,
	validation_data=validation_generator,
	nb_val_samples=len(validation_samples)*6,
	nb_epoch=15,
)

# Save the model when training session is finished.

model.save('model.h5')

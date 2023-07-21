import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from tensorflow.keras import layers
import tensorflow_addons as tfa
from PIL import Image, ImageDraw
from glob import glob
from scipy.io import loadmat

#from data_process import COCOParser


#2408448
image_size = 224
patch_size = 16
num_patches = (image_size // patch_size) ** 2
projection_dim = 384



BATCH_SIZE = 4
NUM_CLASSES = 20

DATA_DIR = "instance-level-human-parsing/instance-level_human_parsing/instance-level_human_parsing/Training"
NUM_TRAIN_IMAGES = 1000
NUM_VAL_IMAGES = 50

train_images = sorted(glob(os.path.join(DATA_DIR, "Images/*")))[:NUM_TRAIN_IMAGES]
train_masks = sorted(glob(os.path.join(DATA_DIR, "Category_ids/*")))[:NUM_TRAIN_IMAGES]
val_images = sorted(glob(os.path.join(DATA_DIR, "Images/*")))[
    NUM_TRAIN_IMAGES : NUM_VAL_IMAGES + NUM_TRAIN_IMAGES
]
val_masks = sorted(glob(os.path.join(DATA_DIR, "Category_ids/*")))[
    NUM_TRAIN_IMAGES : NUM_VAL_IMAGES + NUM_TRAIN_IMAGES
]





def main(): 

	input_shape = (image_size, image_size, 3)
	learning_rate = 0.001
	weight_decay = 0.0001
	batch_size = 1
	num_epochs = 100
	num_heads = 6
	transformer_units = [projection_dim]
	transformer_layers = 18
	 
	 #data process
	def read_image(image_path, mask=False):
		image = tf.io.read_file(image_path)
		if mask:
			image = tf.image.decode_png(image, channels=1)
			image.set_shape([None, None, 1])
			image = tf.image.resize(images=image, size=[image_size, image_size])
		else:
			image = tf.image.decode_png(image, channels=3)
			image.set_shape([None, None, 3])
			image = tf.image.resize(images=image, size=[image_size, image_size])
			image = tf.keras.applications.resnet50.preprocess_input(image)
		return image
    


	def load_data(image_list, mask_list):
		image = read_image(image_list)
		mask = read_image(mask_list, mask=True)
		return image, mask


	def data_generator(image_list, mask_list):
		dataset = tf.data.Dataset.from_tensor_slices((image_list, mask_list))
		dataset = dataset.map(load_data, num_parallel_calls=tf.data.AUTOTUNE)
		dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)
		return dataset


	train_dataset = data_generator(train_images, train_masks)
	val_dataset = data_generator(val_images, val_masks)
	
	print("\n\n")
	print("Train Dataset:", train_dataset)
	print("Val Dataset:", val_dataset)
	
	
	model = create_UViT(
		input_shape,
		patch_size,
		num_patches,
		projection_dim,
		num_heads,
		transformer_units,
		transformer_layers,
	)
	
	model.summary()
	

	#loss = keras.losses.SparseCategoricalCrossentropy(from_logits=False)
	loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
	model.compile(
		optimizer=keras.optimizers.Adam(learning_rate=0.001),
		loss=loss,
		metrics=["accuracy"],
	)

	
	checkpoint_filepath = "logs/"
	checkpoint_callback = keras.callbacks.ModelCheckpoint(
		checkpoint_filepath,
		monitor="accuracy",
		save_best_only=True,
		save_weights_only=True
	)
	
	
	history = model.fit(train_dataset, validation_data=val_dataset, epochs=25)
		
		

	plt.plot(history.history["loss"])
	plt.title("Training Loss")
	plt.ylabel("loss")
	plt.xlabel("epoch")
	plt.show()

	plt.plot(history.history["accuracy"])
	plt.title("Training Accuracy")
	plt.ylabel("accuracy")
	plt.xlabel("epoch")
	plt.show()

	plt.plot(history.history["val_loss"])
	plt.title("Validation Loss")
	plt.ylabel("val_loss")
	plt.xlabel("epoch")
	plt.show()

	plt.plot(history.history["val_accuracy"])
	plt.title("Validation Accuracy")
	plt.ylabel("val_accuracy")
	plt.xlabel("epoch")
	plt.show()
		
	# Loading the Colormap
	colormap = loadmat(
		"instance-level-human-parsing/instance-level_human_parsing/instance-level_human_parsing/human_colormap.mat"
	)["colormap"]
	colormap = colormap * 100
	colormap = colormap.astype(np.uint8)


	def infer(model, image_tensor):
		predictions = model.predict(np.expand_dims((image_tensor), axis=0))
		predictions = np.squeeze(predictions)
		predictions = np.argmax(predictions, axis=2)
		return predictions


	def decode_segmentation_masks(mask, colormap, n_classes):
		r = np.zeros_like(mask).astype(np.uint8)
		g = np.zeros_like(mask).astype(np.uint8)
		b = np.zeros_like(mask).astype(np.uint8)
		for l in range(0, n_classes):
			idx = mask == l
			r[idx] = colormap[l, 0]
			g[idx] = colormap[l, 1]
			b[idx] = colormap[l, 2]
		rgb = np.stack([r, g, b], axis=2)
		return rgb


	def get_overlay(image, colored_mask):
		image = tf.keras.utils.array_to_img(image)
		image = np.array(image).astype(np.uint8)
		overlay = cv2.addWeighted(image, 0.35, colored_mask, 0.65, 0)
		return overlay


	def plot_samples_matplotlib(display_list, figsize=(5, 3)):
		_, axes = plt.subplots(nrows=1, ncols=len(display_list), figsize=figsize)
		for i in range(len(display_list)):
			if display_list[i].shape[-1] == 3:
				axes[i].imshow(tf.keras.utils.array_to_img(display_list[i]))
			else:
				axes[i].imshow(display_list[i])
		plt.show()


	def plot_predictions(images_list, colormap, model):
		for image_file in images_list:
			image_tensor = read_image(image_file)
			prediction_mask = infer(image_tensor=image_tensor, model=model)
			prediction_colormap = decode_segmentation_masks(prediction_mask, colormap, 20)
			overlay = get_overlay(image_tensor, prediction_colormap)
			plot_samples_matplotlib(
				[image_tensor, overlay, prediction_colormap], figsize=(18, 14)
			)
		
		

	plot_predictions(train_images[:4], colormap, model=model)
			
	
	
	



class Patches(layers.Layer):
	def __init__(self, patch_size):
		super().__init__()
		self.patch_size = patch_size

	def call(self, images):
		batch_size = tf.shape(images)[0]
		patches = tf.image.extract_patches(
			images=images,
			sizes=[1, self.patch_size, self.patch_size, 1],
			strides=[1, self.patch_size, self.patch_size, 1],
			rates=[1, 1, 1, 1],
			padding="VALID",
		)
		patch_dims = patches.shape[-1]
		patches = tf.reshape(patches, [batch_size, -1, patch_dims])
		return patches




class PatchEncoder(layers.Layer):
	def __init__(self, num_patches, projection_dim):
		super().__init__()
		self.num_patches = num_patches
		self.projection = layers.Dense(units=projection_dim)  #embed kortase sob flatten patches gula k (1,3072)->(1,64)
		self.position_embedding = layers.Embedding(			  #and total number of patches is 49								
			input_dim=num_patches, output_dim=projection_dim
		)

	def call(self, patch):
		positions = tf.range(start=0, limit=self.num_patches, delta=1) #start theke delta kore barai limit porjnto number dibe
		encoded = self.projection(patch) + self.position_embedding(positions)  #excluding limit itself
		return encoded
		
		#output dibe (1,49,64). mane 49 ta patch prottekta (1,64) dimension a ase with patch and position embedding
		
		#tahle ViT te input jacche (1,64) er 49 ta tensor



def mlp(x, hidden_units, dropout_rate):
	for units in hidden_units:
		x = layers.Dense(units, activation=tf.nn.gelu)(x)
		x = layers.Dropout(dropout_rate)(x)
	return x
	


def create_UViT(
	input_shape,
	patch_size,
	num_patches,
	projection_dim,
	num_heads,
	transformer_units,
	transformer_layers,
	):
	
	
	inputs = layers.Input(shape=input_shape)
	#creating patches
	patches = Patches(patch_size)(inputs)
	#encoded patches
	encoded_patches = PatchEncoder(num_patches, projection_dim)(patches)
	
	
	
	#creating multiple layer of transformer block
	for _ in range(transformer_layers):
		#normalization layer
		x1 = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
		#multihead-attention layer
		attention_output = layers.MultiHeadAttention(num_heads=num_heads, key_dim=projection_dim, dropout=0.1)(x1, x1)
		#skip conncection
		x2 = layers.Add()([attention_output, encoded_patches])
		#normalization layer
		x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
		#mlp
		x3 = mlp(x3, hidden_units=transformer_units, dropout_rate=0.1)
		#skip conncetion
		encoded_patches = layers.Add()([x3, x2])
		
	
	#FPN	
	encoded_patches = layers.Reshape((14,14,384))(encoded_patches)
	
	encoded_patches = layers.Conv2D(256,(3,3),padding='same')(encoded_patches)
	encoded_patches = layers.UpSampling2D(size=(image_size // 2 // 
	encoded_patches.shape[1], image_size // 2 // encoded_patches.shape[2]),
		interpolation="bilinear",
	)(encoded_patches)
	
	encoded_patches = layers.Conv2D(256,(3,3),padding='same')(encoded_patches)
	encoded_patches = layers.UpSampling2D(size=(image_size // 
	encoded_patches.shape[1], image_size // encoded_patches.shape[2]),
		interpolation="bilinear",
	)(encoded_patches)
	
	encoded_patches = layers.Conv2D(20,(3,3),padding='same')(encoded_patches)

	
	outputs = encoded_patches
		
	return keras.Model(inputs=inputs, outputs=outputs)




if __name__ == '__main__':
	main()



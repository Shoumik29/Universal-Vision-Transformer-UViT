import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from tensorflow.keras import layers
import tensorflow_addons as tfa
from PIL import Image, ImageDraw
from data_process import COCOParser


#2408448
image_size = 896
patch_size = 8
num_patches = (image_size // patch_size) ** 2
projection_dim = 448



train_ann_dir = "coco2017/annotations/instances_train2017.json"
train_img_dir = "coco2017/train2017"

val_img_dir = "coco2017/val2017"
val_ann_dir = "coco2017/annotations/instances_val2017.json"



class My_Custom_Generator(keras.utils.Sequence) :
  
	def __init__(self, image_filenames, batch_size, coco_annotations_file, coco_images_dir) :
		self.image_filenames = image_filenames
		self.batch_size = batch_size
		self.coco_annotations_file = coco_annotations_file
		self.coco_images_dir = coco_images_dir


	def __len__(self) :
		return (np.ceil(len(self.image_filenames) / float(self.batch_size))).astype(np.int)
  
  
	def __getitem__(self, idx) :
		batch_val = self.image_filenames[idx * self.batch_size : (idx+1) * self.batch_size]
		x_val = []
		y_val = []
		
		coco = COCOParser(self.coco_annotations_file, self.coco_images_dir) 
		
		for file_name in batch_val:
			x, y = coco.getting_data(file_name)
			if len(x.shape)<3 or len(y.shape)<3:
				continue
			x_val.append(x)
			y_val.append(y)
		
		return np.array(x_val), np.array(y_val)



def main(): 

	input_shape = (image_size, image_size, 3)
	learning_rate = 0.001
	weight_decay = 0.0001
	batch_size = 1
	num_epochs = 100
	num_heads = 4
	transformer_units = [projection_dim]
	transformer_layers = 18
	 
	#data process		
	total_images_train = sorted(
		[
			os.path.join(train_img_dir, fname)
			for fname in os.listdir(train_img_dir)
			if fname.endswith(".jpg")
		]
	)

	total_images_val = sorted(
		[
			os.path.join(val_img_dir, fname)
			for fname in os.listdir(val_img_dir)
			if fname.endswith(".jpg")
		]
	)
	
				
	my_training_batch_generator = My_Custom_Generator(total_images_train, batch_size, train_ann_dir, train_img_dir)
	my_validation_batch_generator = My_Custom_Generator(total_images_val, batch_size, val_ann_dir, val_img_dir)
	
	
	#history = []
	
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
	

	optimizer = tfa.optimizers.AdamW(learning_rate=learning_rate, weight_decay=weight_decay)
	
	#compile model
	model.compile(optimizer=optimizer, loss=keras.losses.MeanSquaredError())
	
	checkpoint_filepath = "logs/"
	checkpoint_callback = keras.callbacks.ModelCheckpoint(
		checkpoint_filepath,
		monitor="val_loss",
		save_best_only=True,
		save_weights_only=True
	)
	
	model.fit_generator(generator=my_training_batch_generator,
		steps_per_epoch = int(80 // batch_size),
		epochs = 10,
		verbose = 1,
		validation_data = my_validation_batch_generator,
		validation_steps = int(50 // batch_size),
		callbacks = [
		checkpoint_callback,
		keras.callbacks.EarlyStopping(monitor="val_loss", patience=10)])
			
	
	
	



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
		
		
		#output er jhamela main karon projection_dim. ekhane 64 kno dilo????
	
	#segmentation fpn(feature pyramid network) head
	encoded_patches = layers.Reshape((224,224,112))(encoded_patches)
	encoded_patches = layers.Conv2DTranspose(112, (7,7),strides=(2,2),activation="relu", padding="same")(encoded_patches)
	encoded_patches = layers.Conv2DTranspose(64, (7,7),activation="relu", padding="same")(encoded_patches)
	encoded_patches = layers.Conv2DTranspose(32, (7,7),strides=(2,2),activation="relu", padding="same")(encoded_patches)
	encoded_patches = layers.Conv2DTranspose(16, (7,7),activation="relu", padding="same")(encoded_patches)
	encoded_patches = layers.Conv2DTranspose(3, (7,7),activation="relu", padding="same")(encoded_patches)

	outputs = encoded_patches
		
	return keras.Model(inputs=inputs, outputs=outputs)




if __name__ == '__main__':
	main()



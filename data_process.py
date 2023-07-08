from collections import defaultdict
import json
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import os


image_size = 896

# define a list of colors for polygon
color_list = ["pink", "red", "teal", "blue", "orange", "yellow", "black", "magenta","green","aqua"]*10

    
    
class COCOParser:
	def __init__(self, anns_file, imgs_dir):
		with open(anns_file, 'r') as f:
			coco = json.load(f)
                
		self.annIm_dict = defaultdict(list)        
		self.cat_dict = {} 
		self.annId_dict = {}
		self.im_dict = {}
		self.licenses_dict = {}
		for ann in coco['annotations']:           
			self.annIm_dict[ann['image_id']].append(ann) 
			self.annId_dict[ann['id']]=ann
		for img in coco['images']:
			self.im_dict[img['id']] = img
		for cat in coco['categories']:
			self.cat_dict[cat['id']] = cat
		for license in coco['licenses']:
			self.licenses_dict[license['id']] = license
	def get_imgIds(self):
		return list(self.im_dict.keys())
	def get_annIds(self, im_ids):
		im_ids=im_ids if isinstance(im_ids, list) else [im_ids]
		return [ann['id'] for im_id in im_ids for ann in self.annIm_dict[im_id]]
	def load_anns(self, ann_ids):
		im_ids=ann_ids if isinstance(ann_ids, list) else [ann_ids]
		return [self.annId_dict[ann_id] for ann_id in ann_ids]        
	def load_cats(self, class_ids):
		class_ids=class_ids if isinstance(class_ids, list) else [class_ids]
		return [self.cat_dict[class_id] for class_id in class_ids]
	def get_imgLicenses(self,im_ids):
		im_ids=im_ids if isinstance(im_ids, list) else [im_ids]
		lic_ids = [self.im_dict[im_id]["license"] for im_id in im_ids]
		return [self.licenses_dict[lic_id] for lic_id in lic_ids]
		


	def getting_data(self, img_dir):
		
		image = Image.open(img_dir)
		
		img_dir = int(img_dir[19:-4])
		
		np_image = np.array(image)
		
		image = image.resize((image_size, image_size))
		
		x_val = np.array(image)
			
		np_image = np.zeros([np_image.shape[0], np_image.shape[1], 3],dtype=np.uint8)
		image = Image.fromarray(np.uint8(np_image)).convert('RGB')
		ann_ids = self.get_annIds(img_dir)
		annotations = self.load_anns(ann_ids)
    
		#I have modified this part of the code for instance segmentation -shoumik
		for ann in annotations:
			segmentation = ann['segmentation']
			class_id = ann["category_id"]
			class_name = self.load_cats(class_id)[0]["name"]
			color_ = color_list[class_id]
        
        
			draw = ImageDraw.Draw(image)
			for p in segmentation:
				if len(p)>0:
					try:
						draw.polygon((p), fill = color_, outline = color_)
					except:
						pass
		
		image = image.resize((image_size, image_size))
		np_image = np.array(image)
		
		y_val = np_image
		
		return x_val/255., y_val/255.     #normalization helps reducing computing cost
	    	

		
		
		
		
		
		
		
		

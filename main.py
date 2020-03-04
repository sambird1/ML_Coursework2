import os
import matplotlib.pyplot as plt
import json
from PIL import ImageOps, Image
import math
import numpy as np

with open("config.json", 'r') as configfile:
	config = json.load(configfile)

segmentationsFilepath = config['segmentationsFilepath']
CUBFilepath = config['CUBFilepath']
attributesFilepath = config['attributesFilepath']

# create empty dicts to add data to
image_id_summary = {}
class_definitions = {}
parts_definitions = {}
attributedict = {}
attribute_definitions = {}
certainty_definitions = {}

# extract current directory location
current_directory = os.path.dirname(os.path.realpath(__file__))

# function to read string from filename argument, and set extracted values to key argument in the dictionary defined by the dict_type argument
def load_files(filename, key, dict_type):
	inputfilepath = current_directory + filename
	with open (inputfilepath, 'r') as file:
		images = file.read()

	# split at new lines
	splitimages = images.split("\n")

	# iterate through lines (excluding last as each file ends with blank line)
	for idx, item in enumerate(splitimages[:-1], start=1):
		# split at whitespace
		item2 = item.split(" ")
		# set key logic based on source data
		if key == "image_filename":
			dict_type[item2[0]] = {}
			dict_type[item2[0]][key] = item2[1].split("/")[1]
		elif key in ["image_class_label", "train_test_split"]:
			dict_type[item2[0]][key] = item2[1]
		elif key == 'image_bounding_box':
			dict_type[item2[0]][key] = item2[1:]
		elif key in ['class_string', 'part_string', 'attribute_string', 'certainty_string']:
			dict_type[item2[0]] = item2[1]
		elif key in ['image_part_location', 'image_attribute_labels']:
			if item2[1] == '1':
				dict_type[item2[0]][key] = {}
			dict_type[item2[0]][key][item2[1]] = item2[2:]
			# dict_type[item2[0]][key][item2[1]]['is_present'] = item2[2]
			# dict_type[item2[0]][key][item2[1]]['certainty_id'] = item2[3]
			# dict_type[item2[0]][key][item2[1]]['time'] = item2[4]
		elif key == 'class_attribute_certainty':
			attributedict[str(idx)] = {}
			for x in range(0,len(item2)):
				attributedict[str(idx)][str(x+1)] = item2[x]
	return

load_files(CUBFilepath + "images.txt", "image_filename", image_id_summary)
load_files(CUBFilepath + "image_class_labels.txt", "image_class_label", image_id_summary)
load_files(CUBFilepath + "bounding_boxes.txt", "image_bounding_box", image_id_summary)
load_files(CUBFilepath + "train_test_split.txt", "train_test_split", image_id_summary)
load_files(CUBFilepath + "classes.txt", "class_string", class_definitions)
load_files(CUBFilepath + "parts\\part_locs.txt", "image_part_location", image_id_summary)
load_files(CUBFilepath + "parts\\parts.txt", "part_string", parts_definitions)
load_files(CUBFilepath + "attributes\\image_attribute_labels.txt", "image_attribute_labels", image_id_summary)
load_files(CUBFilepath + "attributes\\class_attribute_labels_continuous.txt", "class_attribute_certainty", attributedict)
load_files(CUBFilepath + "attributes\\certainties.txt", "certainty_string", certainty_definitions)
load_files(attributesFilepath + "attributes.txt", "attribute_string", attribute_definitions)

maxwidth = 500
maxheight = 500

# iterate through dict of image_ids
# for x in range(1, 10):
for x in range(1, len(image_id_summary)+1):
	# add class definition to be used as directory to locate individual image files
	for key, value in class_definitions.items():
		if key == image_id_summary[str(x)]['image_class_label']:
			image_id_summary[str(x)]['image_class_string'] = value
	# read in image files
	image_filepath = current_directory + CUBFilepath + "images\\"+ image_id_summary[str(x)]['image_class_string'] + '\\' +image_id_summary[str(x)]['image_filename']
	# image abnd image segment variables commented out as Memory Errors experienced generating original and padded versions of each image
	# image_id_summary[str(x)]['image'] = plt.imread(image_filepath)
	# pad image to standard 500 x 500 pixels size
	with Image.open(image_filepath) as image_jpg:
		image_height = image_jpg.size[0]
		image_width = image_jpg.size[1]
		height_diff = maxheight - image_height
		width_diff = maxwidth - image_width
		pad = (height_diff//2, width_diff//2, math.ceil(height_diff/2), math.ceil(width_diff/2))
		image_id_summary[str(x)]['image_padded'] = np.array(ImageOps.expand(image_jpg, pad))
	# read in segmented image files
	segmentation_filepath = current_directory + segmentationsFilepath + image_id_summary[str(x)]['image_class_string'] + '\\' +image_id_summary[str(x)]['image_filename'][:-4] + ".png"
	# image_id_summary[str(x)]['image_segmentation'] = plt.imread(segmentation_filepath)
	# pad image segemnts to standard 500 x 500 pixels size
	with Image.open(segmentation_filepath) as segment:
		segment_height = segment.size[0]
		segment_width = segment.size[1]
		height_diff = maxheight - segment_height
		width_diff = maxwidth - segment_width
		pad = (height_diff//2, width_diff//2, math.ceil(height_diff/2), math.ceil(width_diff/2))
		image_id_summary[str(x)]['image_segmentation_padded'] = np.array(ImageOps.expand(ImageOps.expand(segment, pad)))

	# append the class level attribute presence value to each attribute
	for key, value in attributedict.items():
		if key == image_id_summary[str(x)]['image_class_label']:
			for y in range(1,len(image_id_summary[str(x)]['image_attribute_labels'])+1):
				image_id_summary[str(x)]['image_attribute_labels'][str(y)].append(value[str(y)])

# print dict for first image_id as an example
print(image_id_summary['1'])

# check length of dict matches number of imageids
print(len(image_id_summary))

# to display image data from arrays
plt.imshow(image_id_summary['1']['image_padded'])
plt.show()
plt.imshow(image_id_summary['1']['image_segmentation_padded'])
plt.show()

train = []
test = []
train_class = {}
test_class = {}
image_sizes_h = []
image_sizes_w = []
box_sizes_h = []
box_sizes_w = []

# split data based on train/test labels
# for x in range(1, 10):
for x in range(1, len(image_id_summary)+1):
	# image_sizes_h.append(image_id_summary[str(x)]['image'].shape[0])
	# image_sizes_w.append(image_id_summary[str(x)]['image'].shape[1])
	box_sizes_h.append(float(image_id_summary[str(x)]['image_bounding_box'][3]))
	box_sizes_w.append(float(image_id_summary[str(x)]['image_bounding_box'][2]))
	if image_id_summary[str(x)]['train_test_split'] == '0':
		train.append(image_id_summary[str(x)])
		if image_id_summary[str(x)]['image_class_label'] in train_class:
			train_class[image_id_summary[str(x)]['image_class_label']] +=1
		else:
			train_class[image_id_summary[str(x)]['image_class_label']] = 1

	else:
		test.append(image_id_summary[str(x)])
		if image_id_summary[str(x)]['image_class_label'] in test_class:
			test_class[image_id_summary[str(x)]['image_class_label']] +=1
		else:
			test_class[image_id_summary[str(x)]['image_class_label']] = 1

# Training Records: 5794
# Test Records: 5994
print("Training Records", len(train))
print("Test Records", len(test))

# check split between test and train by class
for x in range(1,201) :
	print(x, "Train Records: ", train_class[str(x)], "Test Records: ", test_class[str(x)], "Test Proportion: ", test_class[str(x)]/(test_class[str(x)]+train_class[str(x)]), "Train Proportion: ", train_class[str(x)]/(train_class[str(x)]+test_class[str(x)]))

# maximum image height: 497, minimum image width: 500
# print(max(box_sizes_h))
# print(max(box_sizes_w))
# minimum image height: 120, minimum image width: 121
# print(min(image_sizes_h))
# print(min(image_sizes_w))



# TO DO: image_bounding_box and image_part_coordinates, need to be shifted to account for padding


# print(class_definitions)
# print(parts_definitions)
# print(attribute_definitions)
# print(certainty_definitions)


# file not read in:
# CUB_200_2011\\parts\\part_click_locs - summarised in the part_locs data - probably not useful for model training

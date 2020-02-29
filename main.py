import os
from matplotlib import pyplot
import json

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

# iterate through dict of image_ids
for x in range(1, len(image_id_summary)+1):
	# add class definition to be used as directory to locate individual image files
	for key, value in class_definitions.items():
		if key == image_id_summary[str(x)]['image_class_label']:
			image_id_summary[str(x)]['image_class_string'] = value
	# read in image files
	image_filepath = current_directory + CUBFilepath + "images\\"+ image_id_summary[str(x)]['image_class_string'] + '\\' +image_id_summary[str(x)]['image_filename']
	image_id_summary[str(x)]['image'] = pyplot.imread(image_filepath)
	# read in segemented image files
	segmentation_filepath = current_directory + segmentationsFilepath + image_id_summary[str(x)]['image_class_string'] + '\\' +image_id_summary[str(x)]['image_filename'][:-4] + ".png"
	image_id_summary[str(x)]['image_segmentation'] = pyplot.imread(segmentation_filepath)
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
pyplot.imshow(image_id_summary['1']['image'])
pyplot.show()
pyplot.imshow(image_id_summary['1']['image_segmentation'])
pyplot.show()

# print(class_definitions)
# print(parts_definitions)
# print(attribute_definitions)
# print(certainty_definitions)


# file not read in:
# CUB_200_2011\\parts\\part_click_locs - summarised in the part_locs data - probably not useful for model training
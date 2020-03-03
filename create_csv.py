# load all images in a directory
from os import listdir, walk
from matplotlib import image
import pandas as pd

# dirc = []
# rootdir = '/home/c1422205/Documents/Modules/AML/CUB_200_2011/CUB_200_2011/images/'
# 
# for subdir, dirs, files in walk(rootdir):
#     dirc.append(subdir)


# load all images in a directory
loaded_images = list()
for filename in listdir(r'/home/c1422205/Documents/Modules/AML/CUB_200_2011/CUB_200_2011/images/'):
    print(filename)
    img_data = image.imread(r'/home/c1422205/Documents/Modules/AML/CUB_200_2011/CUB_200_2011/images/001.Black_footed_Albatross/' + filename)
    # store loaded image
    loaded_images.append(img_data)
    print('> loaded %s %s' % (filename, img_data.shape))

img_directory = pd.read_csv(r'/home/c1422205/Documents/Modules/AML/CUB_200_2011/CUB_200_2011/images.txt', sep=" ", header=None)
img_directory.columns = ['image_id', 'image_name']

train_test_suggested = pd.read_csv(r'/home/c1422205/Documents/Modules/AML/CUB_200_2011/CUB_200_2011/train_test_split.txt', sep=" ", header=None)
train_test_suggested.columns = ['image_id', 'is_training_image']

bird_class = pd.read_csv(r'/home/c1422205/Documents/Modules/AML/CUB_200_2011/CUB_200_2011/classes.txt', sep=" ", header=None)
bird_class.columns = ['class_id', 'class_name']

#The ground truth class labels (bird species labels) for each image are contained in the file image_class_labels.txt, with each line corresponding to one image
#<image_id> <class_id>
#where <image_id> and <class_id> correspond to the IDs in images.txt and classes.txt, respectively.

# # # # Problem loading: all parts # # # #
#parts = pd.read_csv(r'/home/c1422205/Documents/Modules/AML/CUB_200_2011/CUB_200_2011/parts/parts.txt', sep=" ", header=None)
# parts.columns = ['part_id', 'part_name']
part_loc = pd.read_csv(r'/home/c1422205/Documents/Modules/AML/CUB_200_2011/CUB_200_2011/parts/part_locs.txt', sep=" ", header=None)
part_loc.columns = ['image_id', 'part_id', 'x', 'y', 'visible']


part_click_locs = pd.read_csv(r'/home/c1422205/Documents/Modules/AML/CUB_200_2011/CUB_200_2011/parts/part_click_locs.txt', sep=" ", header=None)
part_click_locs.columns = ['image_id', 'part_id', 'x', 'y', 'visible', 'time']

certainties = pd.read_csv(r'/home/c1422205/Documents/Modules/AML/CUB_200_2011/CUB_200_2011/attributes/certainties.txt', sep=" ", header=None)
certainties.columns = ['certainty_id', 'certainty_name', 'drop']
certainties.drop('drop', axis=1, inplace=True)

# # # # Problem loading: image atrributes # # # #
# image_attribute_labs = pd.read_csv(r'/home/c1422205/Documents/Modules/AML/CUB_200_2011/CUB_200_2011/attributes/image_attribute_labels.txt', sep=" ", header=None)
# image_attribute_labs.columns = ['image_id',  'attribute_id', 'is_present', 'certainty_id', 'time']

class_labs_continuous = pd.read_csv(r'/home/c1422205/Documents/Modules/AML/CUB_200_2011/CUB_200_2011/attributes/class_attribute_labels_continuous.txt', sep=" ", header=None)

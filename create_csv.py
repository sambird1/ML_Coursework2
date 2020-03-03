# load all images in a directory
from os import walk
from matplotlib import image
import pandas as pd



# # # # Birds # # # #
# # # # Bird labels # # # #
img_directory = pd.read_csv(r'/home/c1422205/Documents/Modules/AML/CUB_200_2011/CUB_200_2011/images.txt', sep=" ", header=None)
img_directory.columns = ['image_id', 'image_name']

train_test_suggested = pd.read_csv(r'/home/c1422205/Documents/Modules/AML/CUB_200_2011/CUB_200_2011/train_test_split.txt', sep=" ", header=None)
train_test_suggested.columns = ['image_id', 'is_training_image']

bird_class = pd.read_csv(r'/home/c1422205/Documents/Modules/AML/CUB_200_2011/CUB_200_2011/classes.txt', sep=" ", header=None)
bird_class.columns = ['class_id', 'class_name']

bird_class_labs = pd.read_csv(r'/home/c1422205/Documents/Modules/AML/CUB_200_2011/CUB_200_2011/image_class_labels.txt', sep=" ", header=None)
bird_class_labs.columns = ['image_id', 'class_id']

#Concatenating to csv
bird_lab = img_directory.merge(train_test_suggested, how='left', on='image_id')
bird_lab2 = bird_lab.merge(bird_class_labs, how='left', on='image_id')
bird_lab3 = bird_lab2.merge(bird_class, how='left', on='class_id')



# # # # Bounding Box # # # #
bounding_box = pd.read_csv(r'/home/c1422205/Documents/Modules/AML/CUB_200_2011/CUB_200_2011/bounding_boxes.txt', sep=" ", header=None)
bounding_box.columns = ['image_id', 'x', 'y', 'width', 'height']
bird_labs = bird_lab3.merge(bounding_box, how='left', on='image_id')



# # # # Parts # # # #
parts = pd.read_csv(r'/home/c1422205/Documents/Modules/AML/CUB_200_2011/CUB_200_2011/parts/parts.txt', sep=" ", header=None)
parts.columns = ['part_id', 'part_name']

part_loc = pd.read_csv(r'/home/c1422205/Documents/Modules/AML/CUB_200_2011/CUB_200_2011/parts/part_locs.txt', sep=" ", header=None)
part_loc.columns = ['image_id', 'part_id', 'x', 'y', 'visible']

part_click_locs = pd.read_csv(r'/home/c1422205/Documents/Modules/AML/CUB_200_2011/CUB_200_2011/parts/part_click_locs.txt', sep=" ", header=None)
part_click_locs.columns = ['image_id', 'part_id', 'x', 'y', 'visible', 'time']

#Concatenating to csv

part = part_loc.merge(part_click_locs, how='right', on=['image_id', 'part_id', 'x', 'y', 'visible'])
part1 = part.merge(parts, how='left', on=['part_id'])
part1.to_csv(r'/home/c1422205/Documents/Modules/AML/bird_classification/parts.csv')


# # # # Attributes # # # #
certainties = pd.read_csv(r'/home/c1422205/Documents/Modules/AML/CUB_200_2011/CUB_200_2011/attributes/certainties.txt', sep=" ", header=None)
certainties.columns = ['certainty_id', 'certainty_name', 'drop']
certainties.drop('drop', axis=1, inplace=True)
certainties.to_csv(r'/home/c1422205/Documents/Modules/AML/bird_classification/certainties.csv')

attributes = pd.read_csv(r'/home/c1422205/Documents/Modules/AML/CUB_200_2011/attributes.txt', sep=" ", header=None)
attributes.columns = ['attribute_id', 'attribute_name']

image_attribute_labs = pd.read_csv(r'/home/c1422205/Documents/Modules/AML/CUB_200_2011/CUB_200_2011/attributes/image_attribute_labels.txt', sep=" ", header=None, error_bad_lines=False)
image_attribute_labs.columns = ['image_id',  'attribute_id', 'is_present', 'certainty_id', 'time']
att = attributes.merge(image_attribute_labs, how='right', on='attribute_id')
att.to_csv(r'/home/c1422205/Documents/Modules/AML/bird_classification/attributes.csv')


class_labs_continuous = pd.read_csv(r'/home/c1422205/Documents/Modules/AML/CUB_200_2011/CUB_200_2011/attributes/class_attribute_labels_continuous.txt', sep=" ", header=None)
class_labs_continuous.to_csv(r'/home/c1422205/Documents/Modules/AML/bird_classification/class_labs_continuous.csv')



# # # # Images # # # #
#Taking out subdirectory so only file name exists. Needed to match to image name
bird_labs['image_name'] = bird_labs.image_name.apply(lambda x: x.split(sep='/'))
bird_labs['image_name']= bird_labs.image_name.apply(lambda x: x[1])
bird_labs['image_name']= bird_labs.image_name.apply(lambda x: str(x))

# not enough memory to load all the images so must load in batch_1 first, create csv, then repeat for batch_2 and batch_3 together
batch_1 = []
rootdir = '/home/c1422205/Documents/Modules/AML/CUB_200_2011/CUB_200_2011/images/batch_1'
for subdir, dirs, files in walk(rootdir):
    for file in files:
        print('Loading: ' + file)
        img = image.imread(subdir + '/' + file)
        batch_1.append((file, img))
match1 = pd.DataFrame(batch_1, columns=['image_name', 'img'])
bird_labs_batch = bird_labs.merge(match1, how='left', on='image_name')
bird_labs_batch.to_csv(r'/home/c1422205/Documents/Modules/AML/bird_classification/bird_labs_batch.csv')


#Process batch 2 and 3 together after resetting kernel and loading in previous bird file
from os import walk
from matplotlib import image
import pandas as pd
bird_labs = pd.read_csv(r'/home/c1422205/Documents/Modules/AML/bird_classification/bird_labs_batch.csv', low_memory=False)
bird_labs.drop('Unnamed: 0', axis=1, inplace=True)
batch_2 = []
rootdir = '/home/c1422205/Documents/Modules/AML/CUB_200_2011/CUB_200_2011/images/batch_2/'
for subdir, dirs, files in walk(rootdir):
    for file in files:
        print('Loading: ' + file)
        img = image.imread(subdir + '/' + file)
        batch_2.append((file, img))
match2 = pd.DataFrame(batch_2, columns=['image_name', 'img'])
bird_labs_batch2 = bird_labs.merge(match2, how='left', on='image_name')


batch_3 = []
rootdir = '/home/c1422205/Documents/Modules/AML/CUB_200_2011/CUB_200_2011/images/batch_3/'
for subdir, dirs, files in walk(rootdir):
    for file in files:
        print('Loading: ' + file)
        img = image.imread(subdir + '/' + file)
        batch_3.append((file, img))
match3 = pd.DataFrame(batch_3, columns=['image_name', 'img'])
bird_labs_final = bird_labs_batch2.merge(match3, how='left', on='image_name')

bird_labs_final.img_x.fillna(bird_labs_final.img_y, inplace=True)
bird_labs_final.img_x.fillna(bird_labs_final.img, inplace=True)
bird_labs_final.img_x.isna().sum()
bird_labs_final.rename(columns={'img_x': 'pixels'}, inplace =True)
bird_labs_final.drop(labels=['img_y', 'img'], axis=1, inplace=True)

bird_labs_final.to_csv(r'/home/c1422205/Documents/Modules/AML/bird_classification/bird_labs.csv')


bird_labs = pd.read_csv(r'/home/c1422205/Documents/Modules/AML/bird_classification/bird_labs.csv', low_memory=False)
bird_labs.drop(labels= 'Unnamed: 0', axis=1, inplace=True)
bird_labs.to_csv(r'/home/c1422205/Documents/Modules/AML/bird_classification/bird_labs.csv')

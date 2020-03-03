# # # # Bird labels # # # #
img_directory = pd.read_csv(r'/home/c1422205/Documents/Modules/AML/CUB_200_2011/CUB_200_2011/images.txt', sep=" ", header=None)
img_directory.columns = ['image_id', 'image_name']

train_test_suggested = pd.read_csv(r'/home/c1422205/Documents/Modules/AML/CUB_200_2011/CUB_200_2011/train_test_split.txt', sep=" ", header=None)
train_test_suggested.columns = ['image_id', 'is_training_image']

bird_class = pd.read_csv(r'/home/c1422205/Documents/Modules/AML/CUB_200_2011/CUB_200_2011/classes.txt', sep=" ", header=None)
bird_class.columns = ['class_id', 'class_name']

bird_class_labs = pd.read_csv(r'/home/c1422205/Documents/Modules/AML/CUB_200_2011/CUB_200_2011/image_class_labels.txt', sep=" ", header=None)
bird_class_labs.columns = ['image_id', 'class_id']



# # # # Bounding Box # # # #
bounding_box = pd.read_csv(r'/home/c1422205/Documents/Modules/AML/CUB_200_2011/CUB_200_2011/bounding_boxes.txt', sep=" ", header=None)
bounding_box.columns = ['image_id', 'x', 'y', 'width', 'height']



# # # # Parts # # # #
parts = pd.read_csv(r'/home/c1422205/Documents/Modules/AML/CUB_200_2011/CUB_200_2011/parts/parts.txt', sep=" ", header=None)
parts.columns = ['part_id', 'part_name']

part_loc = pd.read_csv(r'/home/c1422205/Documents/Modules/AML/CUB_200_2011/CUB_200_2011/parts/part_locs.txt', sep=" ", header=None)
part_loc.columns = ['image_id', 'part_id', 'x', 'y', 'visible']

part_click_locs = pd.read_csv(r'/home/c1422205/Documents/Modules/AML/CUB_200_2011/CUB_200_2011/parts/part_click_locs.txt', sep=" ", header=None)
part_click_locs.columns = ['image_id', 'part_id', 'x', 'y', 'visible', 'time']



#Attributes
certainties = pd.read_csv(r'/home/c1422205/Documents/Modules/AML/CUB_200_2011/CUB_200_2011/attributes/certainties.txt', sep=" ", header=None)
certainties.columns = ['certainty_id', 'certainty_name', 'drop']
certainties.drop('drop', axis=1, inplace=True)

# # # # Problem loading: image atrributes # # # #
# image_attribute_labs = pd.read_csv(r'/home/c1422205/Documents/Modules/AML/CUB_200_2011/CUB_200_2011/attributes/image_attribute_labels.txt', sep=" ", header=None)
# image_attribute_labs.columns = ['image_id',  'attribute_id', 'is_present', 'certainty_id', 'time']

class_labs_continuous = pd.read_csv(r'/home/c1422205/Documents/Modules/AML/CUB_200_2011/CUB_200_2011/attributes/class_attribute_labels_continuous.txt', sep=" ", header=None)



# # # # Concatenating # # # #
bird_lab = img_directory.merge(train_test_suggested, how='left', on='image_id')
bird_lab2 = bird_lab.merge(bird_class_labs, how='left', on='image_id')
bird_lab3 = bird_lab2.merge(bird_class, how='left', on='class_id')
bird_lab3.to_csv(r'/home/c1422205/Documents/Modules/AML/bird_labs.csv')


part = part_loc.merge(part_click_locs, how='right', on=['image_id', 'part_id', 'x', 'y', 'visible'])
part1 = part.merge(parts, how='left', on=['part_id'])
part1.to_csv(r'/home/c1422205/Documents/Modules/AML/parts.csv')

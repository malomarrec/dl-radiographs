#cd Documents/Learning/CS231N/CS231N-Project
from PIL import Image
from os import listdir


def get_img_list(path_to_dir, extension=".png"):
	"""
	Collects the filenames of all the files in a directory with a certaine extension
	Input: 
	path_to_dir: path to the directory
	extension
	Output:
	list of file names
	"""
	filenames = listdir(path_to_dir)
	return [filename for filename in filenames if filename.endswith(extension)]





def crop_img(img_path, img_name, out_path):
	""" Rotates and exports images.
	Input: 
	- img_path: path to target image
	- img_name: name of target image
	- out_path: path where to save the result
	- anges [optionnal]: list of totation angles
	"""
	src_im = Image.open(img_path)
	width = src_im.size[0]
	height = src_im.size[1]

	out_img = src_im.crop(
	    (
	    	263,
	        77,
	        width - 213,
	        height - 107
	    )
	)
	out_img.save(out_path + "/" +img_name)



# # data_path = "data/radiographs" #path to directory containing images
# data_path = "data/labels" #path to directory containing images
# out_path = "cropped/labels"
# # out_path = "cropped/radiographs"

# # img_path = "data/test_label.png"
# # image = img_path.split("/")[-1].split(".")[0]
# # crop_img(img_path, image, out_path)

# image_list = get_img_list(data_path)

# for image in image_list:
# 	img_path = data_path + "/" + image
# 	crop_img(img_path, image, out_path)

data_path = "data/raw_radiographs" #path to directory containing images
out_path = "data/radiographs"
# out_path = "cropped/radiographs"

# img_path = "data/test_label.png"
# image = img_path.split("/")[-1].split(".")[0]
# crop_img(img_path, image, out_path)

image_list = get_img_list(data_path)

for image in image_list:
	img_path = data_path + "/" + image
	crop_img(img_path, image, out_path)








from PIL import Image
import numpy as numpy

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

if __name__ == '__main__':
	data_path = "cropped/" #path to directory containing images
	out_path = "images/"

	image_list = get_img_list(data_path)

	for image in image_list:
	# image = 'Radiograph_1_blobs_4.png'
		im = Image.open(data_path + image)
		img_rotate_90 = im.rotate(90)
		img_rotate_180 = im.rotate(180)
		img_rotate_270 = im.rotate(270)

		img_rotate_90.save("90" + image)
		img_rotate_180.save("180" + image)
		img_rotate_270.save("270" + image)
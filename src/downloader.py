import pickle
import urllib.request
import requests # pip install requests --user For downloading, has a few extra thinkgs
import shutil
import zipfile
import tarfile
import os
from glob import glob
import numpy as np
import cv2

# █ The bar

# used this guy https://stackoverflow.com/questions/15644964/python-progress-bar-and-downloads
# - changed a few things
def download_with_progress(link, save_path):
	with open(save_path, 'wb') as f:
		print('Downloading {}'.format(save_path))
		response = requests.get(link, stream=True)
		total_length = response.headers.get('content-length')
		size_KB = int(total_length) / 1024.0 # in KB (Kilo Bytes)
		size_MB = size_KB / 1024.0 # size in MB (Mega Bytes)
		if(total_length is None):
			f.write(response.content)
		else:
			dl = 0
			total_length = int(total_length)

			for data in response.iter_content(chunk_size=4096):
				dl += len(data)
				f.write(data)
				done = (50 * dl // total_length)
				print('\r[{}{}] {:.2f}Mb/{:.2f}Mb'.format('=' * done, ' ' * (50 - done), (dl/1024.0)/1024.0,size_MB), end='')


# extracts a tar file to the destination
def extract(tar_, dest, remove_tar=False):
	tar = tarfile.open(tar_)
	tar.extractall(dest)
	tar.close()

	if(remove_tar):
		os.remove(tar_)


# This function recursively makes directories.
def makedir(directory):
	directory = "".join(str(x) for x in directory)
	try:
		os.stat(directory)
	except:
		try:
			os.mkdir(directory)
		except:
			subDir = directory.split('/')
			while (subDir[-1] == ''):
				subdir = subdir[:-1]
			newDir = ""
			for x in range(len(subDir)-2):
				newDir += (subDir[x])
				newDir += ('/')
			#print ("Attempting to pass... " + str(newDir))
			makedir(newDir)
			os.mkdir(directory)

# CIFAR-10 helpers
# for now just make sure I can work with the datasets
def unpickle(file):
	with open(file, 'rb') as fo:
		dict = pickle.load(fo, encoding='bytes')
	return dict

def convert_images(raw):
	# converts images from the CIFAR-1 format and return a 4-dim array
	float_raw = np.array(raw, dtype=float) / 255.0
	images = float_raw.reshape([-1,3,32,32])
	images = images.transpose([0,2,3,1])
	return images

def get_cifar_10(save_dir='data/cifar-10/'):

	link = 'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
	data_file = 'cifar-10-python.tar.gz'

	download_with_progress(link, data_file)

	# now save the file
	print('\rextracting file...\n', end='')
	makedir('data/')
	makedir(save_dir)
	tar = tarfile.open(data_file)
	tar.extractall(save_dir + '/')
	tar.close()

	os.remove(data_file) # and delete the tar boy
	# now setup the rest of the data...
	
	folder_path = save_dir + 'cifar-10-batches-py/'
	meta = unpickle(folder_path+'batches.meta')[b'label_names']
	names = [x.decode('utf-8') for x in meta]

	# write the classes that are in the dataset to a file
	with open(save_dir + 'classes.txt', 'w') as file:
		for name in names:
			file.write('\'' + name + '\', ')
		file.write('\n')
		file.close()

	start = save_dir 
	# for train
	img_size = 32
	n_channels = 3
	num_classes = 10
	_images_per_file = 10000
	save = start + 'train/'
	makedir(save)
	files = glob(folder_path + 'data_batch_*')
	all_tags = []

	print('\nExtracting Train images....')
	len_files = len(files)

	print(files)
	for idx in range(len_files):
		print('\r{}/{}'.format(idx,len_files), end='')
		file = files[idx].replace('\\','/')
		raw_data = unpickle(file)
		raw_img = raw_data[b'data']

		images = convert_images(raw_img)
		cls = np.array(raw_data[b'labels'])
		filenames = np.array(raw_data[b'filenames'])
		filenames = [x.decode('utf-8') for x in filenames]
		# now save all the images with the label attached
		for x in range(len(images)):
			tag = names[cls[x]] + '+' + filenames[x]
			out = images[x] * 255.0
			cv2.imwrite(save+tag,out)
			all_tags.append(tag)

	with open(save_dir + 'train.lst','w') as file:
		for tag in all_tags:
			file.write('train/'+ tag + '\n')
		file.close()
	
	# now do the test images
	all_tags = []
	raw_data = unpickle(folder_path + 'test_batch')
	raw_img = raw_data[b'data']
	images = convert_images(raw_img)
	cls = np.array(raw_data[b'labels'])
	filenames = np.array(raw_data[b'filenames'])
	filenames = [x.decode('utf-8') for x in filenames]
	save = start + 'test/'
	makedir(save)

	print(len(images))
	print(len(cls))
	print(len(filenames))
	for x in range(len(images)):
		tag = names[cls[x]] + '+' + filenames[x]
		#print(tag)
		#show(images[x])
		out = images[x] * 255.0
		#show(out)
		cv2.imwrite(save+tag,out)
		#print(np.shape(images[x]))
		#input('->')
		all_tags.append(tag)


	with open(save_dir + 'test.lst','w') as file:
		for tag in all_tags:
			file.write('test/' + tag + '\n')

		file.close()#'''
	
	# remove files that are not needed
	shutil.rmtree(folder_path)

	# done

# working on this guy
def get_image_net_2012(save_dir='E:/BINA/image_net_2012/', extract_tar=True):
	if(not os.path.exists(save_dir)): makedir(save_dir)
	test_url = 'http://www.image-net.org/challenges/LSVRC/2012/nnoupb/ILSVRC2012_img_test.tar'
	test_tar = save_dir + 'ILSVRC2012_img_test.tar'
	makedir(save_dir + 'test/')

	train_url = 'http://www.image-net.org/challenges/LSVRC/2012/nnoupb/ILSVRC2012_img_train.tar'
	train_tar = save_dir + 'ILSVRC2012_img_train.tar'
	makedir(save_dir + 'train/')

	val_url = 'http://www.image-net.org/challenges/LSVRC/2012/nnoupb/ILSVRC2012_img_val.tar'
	val_tar = save_dir + 'ILSVRC2012_img_val.tar'
	makedir(save_dir + 'val/')

	# VAL DOWNLOAD
	#download_with_progress(val_url, val_tar)
	# now save the file
	
	# extract val images to the val folder
	#if(extract_tar):
	#	print('\rextracting file...\n', end='')
	#	extract(val_tar, save_dir + 'val/', remove_tar=True)

	# TRAIN DOWNLOAD
	#download_with_progress(train_url, train_tar)
	# now save the file
	
	# extract val images to the val folder
	#if(extract_tar): 
	#	print('\rextracting file...\n', end='')
	#	extract(val_tar, save_dir + 'train/', remove_tar=True)
	# after downloading the train files
	# first lets get all of our tar_paths
	
	if(extract_tar):
		extract(train_tar, save_dir + 'tmp_train/', remove_tar=False)
		train_tars = glob(save_dir + 'train/' + '*.tar')
		# train
		file_len = len(train_tars)
		print('Unpacking {} train tars'.format(file_len))
		for x in range(file_len):
			print('\r{}/{}\t {}'.format(x,file_len, train_tars[x]), end='')
			extract(train_tars[x], save_dir + 'train/', remove_tar=True)

	# TEST DOWNLOAD
	#download_with_progress(test_url, test_tar)
	# now save the file
	#
	# extract val images to the val folder
	if(extract_tar): 
		print('\rextracting file...\n', end='')
		extract(test_tar, save_dir + 'test/', remove_tar=True)

# This guy is for extracting all the tars that are in the image_net_2012 train set
# - each tar is a subset and will be extracted to its own folder
def extract_image_net_train(start_dir='/data1/image_net_2012/train/'):
	tars = glob(start_dir + '*.tar')
	size = len(tars)
	print('Extracting {} folders!'.format(size))

	# just loop through all the paths
	for x in range(size):
		# get some names for the folders
		tar_ = tars[x].replace('\\', '/').split('/')[-1]
		folder_name = tar_.split('.')[0]

		# make a little progress bar too with the current file being extracted displayed
		done = (50 * x // size)
		print('\r[{}{}] {}'.format('=' * done, ' ' * (50 - done), folder_name), end='')
		try: 
			os.makedirs(start_dir + folder_name + '/')
		except FileExistsError as ex: # This is the only error that should be oaky to skip over
			pass

		extract(tars[x], start_dir + folder_name + '/', remove_tar=True)
		

	print('\nDone!')

if __name__ == '__main__':
	#get_cifar_10(save_dir='data/CIFAR-10/')
	#get_image_net_2012(save_dir='/data1/image_net_2012/')
	#extract_image_net_train()
	print('extracting val...')
	extract('/data1/image_net_2012/ILSVRC2012_img_val.tar', '/data1/image_net_2012/val/', remove_tar=True)
	
	print('extracting test...')
	extract('/data1/image_net_2012/ILSVRC2012_img_test.tar', '/data1/image_net_2012/test/', remove_tar=True)
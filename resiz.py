from PIL import Image
import os

'''
making each image of shape 28X28 and combining both hindi end english images
class wise.
'''

def resize_img(ip_path,op_path):
	basewidth = 28
	img = Image.open(ip_path)
	wpercent = (basewidth/float(img.size[0]))
	hsize = int((float(img.size[1])*float(wpercent)))
	img = img.resize((basewidth,hsize), Image.ANTIALIAS)
	img.save(op_path) 

for i in range(0,10):
	for p in os.listdir(f'english/test/class{i}'):
		path=f'english/test/class{i}/'+p
		op_path='data/c'+str(i)+'/'+p
		print(op_path)
		resize_img(path,op_path)

	for p in os.listdir(f'english/train/class{i}'):
		path=f'english/train/class{i}/'+p
		op_path='data/c'+str(i)+'/'+p
		resize_img(path,op_path)
		

for i in range(0,10):
	for p in os.listdir(f'hindi/Test/digit_{i}'):
		path=f'hindi/Test/digit_{i}/'+p
		op_path='data/c'+str(i)+'/'+p
		resize_img(path,op_path)

	for p in os.listdir(f'hindi/Train/digit_{i}'):
		path=f'hindi/Train/digit_{i}/'+p
		op_path='data/c'+str(i)+'/'+p
		resize_img(path,op_path)
		


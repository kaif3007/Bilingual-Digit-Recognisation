import shutil
import os


'''
used for splitting
train 7000 per class
test 1000 per class
validate 1000 per class

train 1400 hindi  5600 english
test and validate 300 hindi and 700 english

'''
for i in range(0,10):
	cnt=0
	for p in os.listdir(f'english/test/class{i}'):
		if cnt>=700:
			break
		path=f'data1/c{i}/'+p
		if not os.path.exists(path):
			continue
		print(path)
		cnt+=1
		shutil.move(path,f'data1/test/c{i}')


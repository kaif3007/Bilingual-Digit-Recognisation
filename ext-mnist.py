
import os
import imageio
import tensorflow as tf


output_dir = "/home/kaif/Desktop/soumen-project"

mnist = tf.keras.datasets.mnist
(train_images, train_labels),(test_images, test_labels) = mnist.load_data()


def save_class0(save_dir,image,i):
    class_dir = os.path.join(output_dir, save_dir +'/class0')
    file_name = str(class_dir + '/' + str(i) + '.jpg')
    if not os.path.exists(class_dir):
        os.mkdir(class_dir)
        imageio.imwrite(file_name,image)
    else:
        imageio.imwrite(file_name,image)

def save_class1(save_dir,image,i):
    class_dir = os.path.join(output_dir, save_dir +'/class1')
    file_name = str(class_dir + '/' + str(i) + '.jpg')
    if not os.path.exists(class_dir):
        os.mkdir(class_dir)
        imageio.imwrite(file_name,image)
    else:
        imageio.imwrite(file_name,image)
  
def save_class2(save_dir,image,i):
    class_dir = os.path.join(output_dir, save_dir +'/class2')
    file_name = str(class_dir + '/' + str(i) + '.jpg')
    if not os.path.exists(class_dir):
        os.mkdir(class_dir)
        imageio.imwrite(file_name,image)
    else:
        imageio.imwrite(file_name,image)

def save_class3(save_dir,image,i):
    class_dir = os.path.join(output_dir, save_dir +'/class3')
    file_name = str(class_dir + '/' + str(i) + '.jpg')
    if not os.path.exists(class_dir):
        os.mkdir(class_dir)
        imageio.imwrite(file_name,image)
    else:
        imageio.imwrite(file_name,image)


def save_class4(save_dir,image,i):
    class_dir = os.path.join(output_dir, save_dir +'/class4')
    file_name = str(class_dir + '/' + str(i) + '.jpg')
    if not os.path.exists(class_dir):
        os.mkdir(class_dir)
        imageio.imwrite(file_name,image)
    else:
        imageio.imwrite(file_name,image)

def save_class5(save_dir,image,i):
    class_dir = os.path.join(output_dir, save_dir +'/class5')
    print(os.path.exists(output_dir))
    file_name = str(class_dir + '/' + str(i) + '.jpg')
    if not os.path.exists(class_dir):
        print("kaif")
        os.mkdir(class_dir)
        print(file_name)
        imageio.imwrite(file_name,image)
    else:
        imageio.imwrite(file_name,image)
 
def save_class6(save_dir,image,i):
    class_dir = os.path.join(output_dir, save_dir +'/class6')
    file_name = str(class_dir + '/' + str(i) + '.jpg')
    if not os.path.exists(class_dir):
        os.mkdir(class_dir)
        imageio.imwrite(file_name,image)
    else:
        imageio.imwrite(file_name,image)

def save_class7(save_dir,image,i):
    class_dir = os.path.join(output_dir, save_dir +'/class7')
    file_name = str(class_dir + '/' + str(i) + '.jpg')
    if not os.path.exists(class_dir):
        os.mkdir(class_dir)
        imageio.imwrite(file_name,image)
    else:
        imageio.imwrite(file_name,image)
 
 
def save_class8(save_dir,image,i):
    class_dir = os.path.join(output_dir, save_dir +'/class8')
    file_name = str(class_dir + '/' + str(i) + '.jpg')
    if not os.path.exists(class_dir):
        os.mkdir(class_dir)
        imageio.imwrite(file_name,image)
    else:
        imageio.imwrite(file_name,image)
        

def save_class9(save_dir,image,i):
    class_dir = os.path.join(output_dir, save_dir +'/class9')
    file_name = str(class_dir + '/' + str(i) + '.jpg')
    if not os.path.exists(class_dir):
        os.mkdir(class_dir)
        imageio.imwrite(file_name,image)
    else:
        imageio.imwrite(file_name,image)

# dict instead of switch case or if else technique
class_label = {
        0: save_class0,
        1: save_class1,
        2: save_class2,
        3: save_class3,
        4: save_class4,
        5: save_class5,
        6: save_class6,
        7: save_class7,
        8: save_class8,
        9: save_class9
        }
  
# saving training data
i = 0
num_images = len(train_images)
print(num_images)
for i in range(0, num_images):
        
    image = train_images[i]
    image = image.reshape(28,28)
    label = train_labels[i]
    
    class_label[label]('train',image,i) # call dict as method
    i += 1
    
 
# saving test data
i = 0
num_images = len(test_images)
for i in range(0, num_images):
        
    image = test_images[i]
 
    image = image.reshape(28,28)
    label = test_labels[i]
    
    class_label[label]('test',image,i) # call dict as method
    i += 1

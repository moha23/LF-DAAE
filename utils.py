import tensorflow as tf
import numpy as np
import os 

def make_mat(path1):
    """Read folder paths to a list"""
    Data_folder = sorted(os.listdir(path1))
    Data_path_mat = []
    for f in Data_folder:
        if not f.startswith('.'):
            Data_path_mat.append(path1 + '/' + f)
    return(Data_path_mat)

def select_views(x_train_images_path, row):
    """Select views belonging to given row."""
    finalViews =[]
    for view in x_train_images_path:
        name = os.path.basename(view)
        namelist = name.split('_')
        if namelist[0] == str(row):
            finalViews.append(view)
    finalViews = sorted(finalViews, key=lambda f: int((f.split('_')[1]).split('.')[0]))
    return(finalViews)

def read_image(filename):
    """Loads and returns a PNG image file with values in [0..1]."""
    string = tf.io.read_file(filename)
    image = tf.image.decode_image(string, channels=3)
    image = tf.cast(image, tf.float32)
    image /= 255
    return(image)

def quantize_image(image):
    """Convert [0..1] float image to [0..255] uint8."""
    image = tf.round(image * 255)
    image = tf.saturate_cast(image, tf.uint8)
    return image

def read_image_test(path1):
    """Loads and returns a PNG image file with values in [0..1]."""
    img = read_image(path1)
    [h,w,c]=np.shape(img)
    if h!=512:
        if h!=434:
            if (w-528)%2==0:
                img=img[:,6:-6,:]
            else:
                img=img[:,7:-6,:]
            if (h-352)%2==0:
                img=img[12:-12,:,:]
            else:
                img=img[12:-11,:,:]
        else:
            img=img[1:-1,1:,:]
    return(img)

def save_img(path,org,img,row,col):
    """Saves an image to a PNG file."""
    img = tf.squeeze(quantize_image(img))
    if org:
        filename=path+'/org_'+str(row)+'_'+str(col)+'.png'
    else:
        filename=path+'/rec_'+str(row)+'_'+str(col)+'.png'
    string = tf.image.encode_png(img)
    return tf.io.write_file(filename, string)

def preprocess_data(images_path):
    """Read data and create feature tensors."""
    for i in range(8):
        img = tf.expand_dims(read_image(images_path[i]),axis=0)
        if i==0:
            images = img
        else:
            images = tf.concat((images,img),axis=0)
    row=tf.strings.to_number(tf.strings.split(tf.strings.split(images_path[0],'_' )[0],'/')[-1])
    pos_x=tf.math.divide(row*tf.ones([64,64,3]),5)
    batchfeature1 = tf.stack((images[0,:,:,:],images[5,:,:,:],pos_x,(1/5)*tf.ones([64,64,3])),axis=0)
    batchfeature2 = tf.stack((images[1,:,:,:],images[5,:,:,:],pos_x,(2/5)*tf.ones([64,64,3])),axis=0)
    batchfeature3 = tf.stack((images[2,:,:,:],images[5,:,:,:],pos_x,(3/5)*tf.ones([64,64,3])),axis=0)
    batchfeature4 = tf.stack((images[3,:,:,:],images[5,:,:,:],pos_x,(4/5)*tf.ones([64,64,3])),axis=0)
    batchfeature5 = tf.stack((images[4,:,:,:],images[5,:,:,:],pos_x,(5/5)*tf.ones([64,64,3])),axis=0)
    batchfeature6 = tf.stack((images[5,:,:,:],images[5,:,:,:],pos_x,(6/5)*tf.ones([64,64,3])),axis=0)
    batchfeature7 = tf.stack((images[6,:,:,:],images[5,:,:,:],pos_x,(7/5)*tf.ones([64,64,3])),axis=0)
    batchfeature8 = tf.stack((images[7,:,:,:],images[5,:,:,:],pos_x,(8/5)*tf.ones([64,64,3])),axis=0)
    return(images,batchfeature1,batchfeature2,batchfeature3,batchfeature4,batchfeature5,batchfeature6,batchfeature7,batchfeature8)

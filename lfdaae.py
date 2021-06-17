"""

LF-DAAE : Disparity-Aware AutoEncoder for Light Field 
Image Compression

This is the source code to  light field image compression model 
published in:

M. Singh and R. M. Rameshan
"Learning-Based Practical Light Field Image Compression 
Using A Disparity-Aware Model"
Picture Coding Symposium (PCS), 2021

"""
import argparse
import functools
import glob
import sys
import os
import random
from absl import app
from absl.flags import argparse_flags
from joblib import Parallel, delayed
import multiprocessing
import tensorflow as tf
import tensorflow_compression as tfc
from tensorflow_addons.image import dense_image_warp
import numpy as np
from utils import *

tf.random.set_seed(0)
random.seed(0)
np.random.seed(0)

class ColorAnalysisTransform(tf.keras.Sequential):
    """The analysis transform for color module."""
    def __init__(self):
        super().__init__()
        conv = functools.partial(tfc.SignalConv3D, corr=True, strides_down=2,
                                 padding="same_zeros", use_bias=True)
        layers = [
            conv(192, (3, 3, 3), name="layer_0", activation=tfc.GDN(name="gdn_0")),
            conv(192, (3, 3, 3), name="layer_1", activation=tfc.GDN(name="gdn_1")),
            conv(192, (3, 3, 3), name="layer_2", activation=tfc.GDN(name="gdn_2")),
            conv(320, (3, 3, 3), name="layer_3", activation=None),
        ]
        for layer in layers:
            self.add(layer)

class ColorSynthesisTransform(tf.keras.Sequential):
    """The synthesis transform for color module."""
    def __init__(self):
        super().__init__()
        conv = functools.partial(tfc.SignalConv3D, corr=False, 
                                 padding="same_zeros", use_bias=True)
        layers = [
            conv(192, (3, 3, 3), name="layer_0",strides_up=(2,2,2),
                 activation=tfc.GDN(name="igdn_0", inverse=True)),
            conv(192, (3, 3, 3), name="layer_1",strides_up=(2,2,2),
                 activation=tfc.GDN(name="igdn_1", inverse=True)),
            conv(192, (3, 3, 3), name="layer_2",strides_up=(2,2,2),
                 activation=tfc.GDN(name="igdn_2", inverse=True)),
            conv(3, (3, 3, 3), name="layer_3",strides_up=(1,2,2),
                 activation=None),
        ]
        for layer in layers:
            self.add(layer)

class DispAnalysisTransform(tf.keras.Sequential):
    """The analysis transform for disparity modules."""
    def __init__(self):
        super().__init__()
        conv = functools.partial(tfc.SignalConv3D, corr=True,
                                 padding="same_zeros", use_bias=True)
        layers = [
            conv(16, (3, 3, 3), name="layer_0", 
                activation=tfc.GDN(name="gdn_0")),
            conv(16, (3, 3, 3), name="layer_1", strides_down=(2,2,2), 
                  activation=tfc.GDN(name="gdn_1")),
            conv(8, (3, 3, 3), name="layer_2", strides_down=(2,4,4), 
                  activation=None),
        ]
        for layer in layers:
            self.add(layer)
            
class DispSynthesisTransform(tf.keras.Sequential):
    """The synthesis transform for disparity module."""

    def __init__(self):
        super().__init__()
        conv = functools.partial(tfc.SignalConv3D, corr=False,
                                 padding="same_zeros", use_bias=True)
        layers = [
            conv(8, (3, 3, 3), name="layer_0", strides_up=(1,4,4),
                 activation=tfc.GDN(name="igdn_0", inverse=True)),
            conv(8, (3, 3, 3), name="layer_1", strides_up=(1,2,2),
                 activation=tfc.GDN(name="igdn_1", inverse=True)),
            conv(2, (3, 3, 3), name="layer_2",
                 activation=None),
        ]
        for layer in layers:
            self.add(layer)


class CompressionModel(tf.Module):
    """Module that encapsulates the compression model."""

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.color_analysis_transform = ColorAnalysisTransform()
        self.disp1_analysis_transform = DispAnalysisTransform()
        self.disp2_analysis_transform = DispAnalysisTransform()
        self.disp3_analysis_transform = DispAnalysisTransform()
        self.disp4_analysis_transform = DispAnalysisTransform()
        self.disp5_analysis_transform = DispAnalysisTransform()
        self.disp6_analysis_transform = DispAnalysisTransform()
        self.disp7_analysis_transform = DispAnalysisTransform()
        self.disp8_analysis_transform = DispAnalysisTransform()

        self.color_synthesis_transform = ColorSynthesisTransform()
        self.disp1_synthesis_transform = DispSynthesisTransform()
        self.disp2_synthesis_transform = DispSynthesisTransform()
        self.disp3_synthesis_transform = DispSynthesisTransform()
        self.disp4_synthesis_transform = DispSynthesisTransform()
        self.disp5_synthesis_transform = DispSynthesisTransform()
        self.disp6_synthesis_transform = DispSynthesisTransform()
        self.disp7_synthesis_transform = DispSynthesisTransform()
        self.disp8_synthesis_transform = DispSynthesisTransform()
        
        
        self.entropy_bottleneck_color = tfc.NoisyDeepFactorized(batch_shape=[320])
        self.entropy_bottleneck_disp = tfc.NoisyDeepFactorized(batch_shape=[8])

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)

        tf.summary.experimental.set_step(self.optimizer.iterations)
        self.writer = tf.summary.create_file_writer(self.args.checkpoint_dir)
        
        
    def train_one_step(self,x,f1,f2,f3,f4,f5,f6,f7,f8):
        """Build model and apply gradients."""
        with tf.GradientTape() as tape, self.writer.as_default():
            loss,total_bpp,total_mse = self._run("train", x=x,feature1=f1,feature2=f2,feature3=f3,feature4=f4,
                                                  feature5=f5,feature6=f6,feature7=f7,feature8=f8)
        grads = tape.gradient(loss, self.trainable_variables)
        norm = tf.constant(1.0, dtype=tf.float32)
        grads = [tf.clip_by_norm(g, norm) for g in grads]
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        self.writer.flush()
        return loss,total_bpp,total_mse
    
    def compress(self, x,f1,f2,f3,f4,f5,f6,f7,f8,outputfile,path,row):
        """Build model and compress latents."""
        mse, bpp, x_hat, pack = self._run("compress", x=x,feature1=f1,feature2=f2,feature3=f3,feature4=f4,
                                            feature5=f5,feature6=f6,feature7=f7,feature8=f8)

        # Write a binary file with the shape information and the compressed string.
        packed = tfc.PackedTensors()
        tensors, arrays = zip(*pack)
        packed.pack(tensors, arrays)
        with open(outputfile, "wb") as f:
            f.write(packed.string)

        x *= 255  # x_hat is already in the [0..255] range
        psnr = tf.squeeze(tf.image.psnr(x_hat, x, 255))
        msssim = tf.squeeze(tf.image.ssim_multiscale(x_hat, x, 255))

        # The actual bits per pixel including overhead.
        x_shape = tf.shape(x)
        num_pixels = tf.cast(tf.reduce_prod(x_shape[:-1]), dtype=tf.float32)
        packed_bpp = len(packed.string) * 8 / num_pixels
        
        for col in range(np.shape(x_hat)[1]):
            img = x_hat[0,col,:,:,:]/255 
            save_img(path,0,img,row,col+1)
        return x_hat, psnr, msssim, packed_bpp

    def decompress(self, bit_strings):
        """Build model and decompress bitstrings to generate a reconstruction."""
        return self._run("decompress", bit_strings=bit_strings)
    

    def _run(self, mode, x=None,feature1=None,feature2=None,feature3=None,feature4=None,
        feature5=None,feature6=None,feature7=None,feature8=None,bit_strings=None):
        """Run model according to `mode` (train, compress, or decompress)."""
        training = (mode == "train")

        if mode == "decompress":
            x_shape,x_encoded_shape,disp_encoded_shape = bit_strings[:3]
            x_encoded_string,disp1_encoded_string,disp2_encoded_string = bit_strings[3:6]
            disp3_encoded_string,disp4_encoded_string,disp5_encoded_string = bit_strings[6:9]
            disp6_encoded_string,disp7_encoded_string,disp8_encoded_string = bit_strings[9:]

        else:
            x_shape = tf.shape(x)[1:-1]

            # Build the encoder (analysis) half of the color module.
            x_encoded = self.color_analysis_transform(x)

            # Build the encoder (analysis) half of the disparity module.
            disp1_encoded = self.disp1_analysis_transform(feature1)
            disp2_encoded = self.disp2_analysis_transform(feature2)
            disp3_encoded = self.disp3_analysis_transform(feature3)
            disp4_encoded = self.disp4_analysis_transform(feature4)
            disp5_encoded = self.disp5_analysis_transform(feature5)
            disp6_encoded = self.disp6_analysis_transform(feature6)
            disp7_encoded = self.disp7_analysis_transform(feature7) 
            disp8_encoded = self.disp8_analysis_transform(feature8)

            x_encoded_shape = tf.shape(x_encoded)[1:-1]
            disp_encoded_shape = tf.shape(disp1_encoded)[1:-1]

        if mode == "train":
            num_pixels = tf.cast(self.args.batch_size * 8 * 64 ** 2, tf.float32)
            num_pixels_disp = num_pixels/2.0
        else:
            num_pixels = tf.cast(tf.reduce_prod(x_shape), tf.float32)
            num_pixels_disp = num_pixels/2.0

        # Build the entropy models for the latents.
        em_color = tfc.ContinuousBatchedEntropyModel(
            self.entropy_bottleneck_color, coding_rank=4,
            compression=not training, no_variables=True)
        em_disp = tfc.ContinuousBatchedEntropyModel(
            self.entropy_bottleneck_disp, coding_rank=4,
            compression=not training, no_variables=True)

        if mode != "decompress":
            # When training, *_bpp is based on the noisy version of the latents.
            _, x_encoded_bits = em_color(x_encoded, training=training)
            _, disp1_encoded_bits = em_disp(disp1_encoded, training=training)
            _, disp2_encoded_bits = em_disp(disp2_encoded, training=training)
            _, disp3_encoded_bits = em_disp(disp3_encoded, training=training)
            _, disp4_encoded_bits = em_disp(disp4_encoded, training=training)
            _, disp5_encoded_bits = em_disp(disp5_encoded, training=training)
            _, disp6_encoded_bits = em_disp(disp6_encoded, training=training)
            _, disp7_encoded_bits = em_disp(disp7_encoded, training=training)
            _, disp8_encoded_bits = em_disp(disp8_encoded, training=training)
            x_encoded_bpp = tf.reduce_mean(x_encoded_bits) / num_pixels
            disp1_encoded_bpp = tf.reduce_mean(disp1_encoded_bits) / num_pixels_disp
            disp2_encoded_bpp = tf.reduce_mean(disp2_encoded_bits) / num_pixels_disp
            disp3_encoded_bpp = tf.reduce_mean(disp3_encoded_bits) / num_pixels_disp
            disp4_encoded_bpp = tf.reduce_mean(disp4_encoded_bits) / num_pixels_disp
            disp5_encoded_bpp = tf.reduce_mean(disp5_encoded_bits) / num_pixels_disp
            disp6_encoded_bpp = tf.reduce_mean(disp6_encoded_bits) / num_pixels_disp
            disp7_encoded_bpp = tf.reduce_mean(disp7_encoded_bits) / num_pixels_disp
            disp8_encoded_bpp = tf.reduce_mean(disp8_encoded_bits) / num_pixels_disp
            total_bpp = x_encoded_bpp+disp1_encoded_bpp+disp2_encoded_bpp+disp3_encoded_bpp+ \
            disp4_encoded_bpp+disp5_encoded_bpp+disp6_encoded_bpp+disp7_encoded_bpp+disp8_encoded_bpp


        if training:
            # Use rounding (instead of uniform noise) to modify latents before passing them
            # to their respective synthesis transforms. Note that quantize() overrides the
            # gradient to create a straight-through estimator.
            x_encoded_hat = em_color.quantize(x_encoded)
            disp1_encoded_hat = em_disp.quantize(disp1_encoded)
            disp2_encoded_hat = em_disp.quantize(disp2_encoded)
            disp3_encoded_hat = em_disp.quantize(disp3_encoded)
            disp4_encoded_hat = em_disp.quantize(disp4_encoded)
            disp5_encoded_hat = em_disp.quantize(disp5_encoded)
            disp6_encoded_hat = em_disp.quantize(disp6_encoded)
            disp7_encoded_hat = em_disp.quantize(disp7_encoded)
            disp8_encoded_hat = em_disp.quantize(disp8_encoded)
            x_encoded_string = None
            disp1_encoded_string = None
            disp2_encoded_string = None
            disp3_encoded_string = None
            disp4_encoded_string = None
            disp5_encoded_string = None
            disp6_encoded_string = None
            disp7_encoded_string = None
            disp8_encoded_string = None
        else:
            if mode == "compress":
                x_encoded_string = em_color.compress(x_encoded)
                disp1_encoded_string = em_disp.compress(disp1_encoded)
                disp2_encoded_string = em_disp.compress(disp2_encoded)
                disp3_encoded_string = em_disp.compress(disp3_encoded)
                disp4_encoded_string = em_disp.compress(disp4_encoded)
                disp5_encoded_string = em_disp.compress(disp5_encoded)
                disp6_encoded_string = em_disp.compress(disp6_encoded)
                disp7_encoded_string = em_disp.compress(disp7_encoded)
                disp8_encoded_string = em_disp.compress(disp8_encoded)
            x_encoded_hat = em_color.decompress(x_encoded_string, x_encoded_shape)
            disp1_encoded_hat = em_disp.decompress(disp1_encoded_string, disp_encoded_shape)
            disp2_encoded_hat = em_disp.decompress(disp2_encoded_string, disp_encoded_shape)
            disp3_encoded_hat = em_disp.decompress(disp3_encoded_string, disp_encoded_shape)
            disp4_encoded_hat = em_disp.decompress(disp4_encoded_string, disp_encoded_shape)
            disp5_encoded_hat = em_disp.decompress(disp5_encoded_string, disp_encoded_shape)
            disp6_encoded_hat = em_disp.decompress(disp6_encoded_string, disp_encoded_shape)
            disp7_encoded_hat = em_disp.decompress(disp7_encoded_string, disp_encoded_shape)
            disp8_encoded_hat = em_disp.decompress(disp8_encoded_string, disp_encoded_shape)

        # Build the decoder (synthesis) half of the color module.
        x_tilde = self.color_synthesis_transform(x_encoded_hat)

        # Build the decoder (synthesis) half of the disparity module.
        dispmap1 = tf.squeeze(self.disp1_synthesis_transform(disp1_encoded_hat),axis=1)
        dispmap2 = tf.squeeze(self.disp2_synthesis_transform(disp2_encoded_hat),axis=1)
        dispmap3 = tf.squeeze(self.disp3_synthesis_transform(disp3_encoded_hat),axis=1)
        dispmap4 = tf.squeeze(self.disp4_synthesis_transform(disp4_encoded_hat),axis=1)
        dispmap5 = tf.squeeze(self.disp5_synthesis_transform(disp5_encoded_hat),axis=1)
        dispmap6 = tf.squeeze(self.disp6_synthesis_transform(disp6_encoded_hat),axis=1)
        dispmap7 = tf.squeeze(self.disp7_synthesis_transform(disp7_encoded_hat),axis=1)
        dispmap8 = tf.squeeze(self.disp8_synthesis_transform(disp8_encoded_hat),axis=1)

        # Perform warping with disparity maps of respective slices of x_tilde.
        x_hat1 = dense_image_warp(x_tilde[:,0,:,:,:],dispmap1)        
        x_hat2 = dense_image_warp(x_tilde[:,1,:,:,:],dispmap2)       
        x_hat3 = dense_image_warp(x_tilde[:,2,:,:,:],dispmap3)       
        x_hat4 = dense_image_warp(x_tilde[:,3,:,:,:],dispmap4)        
        x_hat5 = dense_image_warp(x_tilde[:,4,:,:,:],dispmap5)        
        x_hat6 = dense_image_warp(x_tilde[:,5,:,:,:],dispmap6)        
        x_hat7 = dense_image_warp(x_tilde[:,6,:,:,:],dispmap7)        
        x_hat8 = dense_image_warp(x_tilde[:,7,:,:,:],dispmap8)
        x_hat = tf.stack([x_hat1,x_hat2,x_hat3,x_hat4,x_hat5,x_hat6,x_hat7,x_hat8], axis = 1)

        # Mean squared error across pixels.
        if training:
            # Don't clip or round pixel values while training.
            mse = tf.reduce_mean(tf.math.squared_difference(x, x_hat))
            mse *= 255 ** 2  # multiply by 255^2 to correct for rescaling
        else:
            x_hat = tf.clip_by_value(x_hat, 0, 1)
            x_hat = tf.round(x_hat * 255)
            if mode == "compress":
                mse = tf.reduce_mean(tf.math.squared_difference(x * 255, x_hat))
        if mode == "train":
            # Calculate and return the rate-distortion loss: R + lmbda * D.
            loss = total_bpp + self.args.lmbda * mse
            return loss,total_bpp,mse
        elif mode == "compress":
            # Create `pack` dict mapping tensors to values.
            tensors = [x_shape,x_encoded_shape,disp_encoded_shape,x_encoded_string,disp1_encoded_string,
                      disp2_encoded_string,disp3_encoded_string,disp4_encoded_string,disp5_encoded_string,
                      disp6_encoded_string,disp7_encoded_string,disp8_encoded_string]
            pack = [(v, v.numpy()) for v in tensors]
            return mse, total_bpp, x_hat, pack
        elif mode == "decompress":
            return x_hat


def compress(args):
    """Compresses a light field image."""
    # This is the sequential version, processing 8 rows one after another.
    x_folders_path =  args.input_file
    x_test_images_path = np.asarray(make_mat(x_folders_path))
    # Build model, restore optimized parameters.
    model = CompressionModel(args)
    checkpoint = tf.train.Checkpoint(model=model)
    restore_path = tf.train.latest_checkpoint(args.checkpoint_dir)
    checkpoint.restore(restore_path)
    # Read LF image rows and create feature tensors.
    for row in range(8):                  
        x_val_images_path_curr = select_views(x_test_images_path, row+1)       
        for img in range(8):
            if img == 0:
                images = np.expand_dims(read_image_test(x_val_images_path_curr[img]),axis = 0)
            else:
                temp = np.expand_dims(read_image_test(x_val_images_path_curr[img]),axis = 0)
                images = np.concatenate((images, temp),axis = 0)
        pos_x = ((row+1)/5.0) * np.ones(np.shape(images[0,:,:,:]))
        images = np.expand_dims(images, axis = 0)
        pos_x = np.expand_dims(pos_x, axis = 0) 
        pos_y =  np.ones(np.shape(images[:,0,:,:,:]))
        batch_feature1 = np.stack((images[:,0,:,:,:],images[:,4,:,:,:],pos_x,(1.0/5.0)*pos_y), axis=1)
        batch_feature2 = np.stack((images[:,1,:,:,:],images[:,4,:,:,:],pos_x,(2.0/5.0)*pos_y), axis=1)
        batch_feature3 = np.stack((images[:,2,:,:,:],images[:,4,:,:,:],pos_x,(3.0/5.0)*pos_y), axis=1)
        batch_feature4 = np.stack((images[:,3,:,:,:],images[:,4,:,:,:],pos_x,(4.0/5.0)*pos_y), axis=1)
        batch_feature5 = np.stack((images[:,4,:,:,:],images[:,4,:,:,:],pos_x,(5.0/5.0)*pos_y), axis=1)
        batch_feature6 = np.stack((images[:,5,:,:,:],images[:,4,:,:,:],pos_x,(6.0/5.0)*pos_y), axis=1)
        batch_feature7 = np.stack((images[:,6,:,:,:],images[:,4,:,:,:],pos_x,(7.0/5.0)*pos_y), axis=1)
        batch_feature8 = np.stack((images[:,7,:,:,:],images[:,4,:,:,:],pos_x,(8.0/5.0)*pos_y), axis=1)

        if not os.path.exists(args.output_file):
            os.mkdir(args.output_file)
        outputfile=args.output_file+'/'+str(row+1)+'.tfci'
        
        # Write the input images as png files.
        for col in range(np.shape(images)[1]):
            inpimg = images[:,col,:,:,:]
            save_img(args.output_file,1,inpimg,row+1,col+1)

        # Compress the LF image rows.
        curr_decoded, psnr, msssim, bpp = model.compress(images,batch_feature1,batch_feature2,
                                                         batch_feature3,batch_feature4,batch_feature5,
                                                         batch_feature6,batch_feature7,batch_feature8,
                                                         outputfile,args.output_file,row+1)
        print("PSNR:%.2f, MS-SSIM:%.2f, BPP:%.2f"%(psnr,msssim,bpp))

def decompress(args):
    """Decompresses a row of LF image."""
    # Three integers for tensor shapes + nine encoded strings.
    np_dtypes = [np.integer] * 3 + [np.bytes_] * 9
    with open(args.input_file, "rb") as f:
        packed = tfc.PackedTensors(f.read())
    arrays = packed.unpack_from_np_dtypes(np_dtypes)

    # Build model and restore optimized parameters.
    model = CompressionModel(args)
    checkpoint = tf.train.Checkpoint(model=model)
    restore_path = tf.train.latest_checkpoint(args.checkpoint_dir)
    checkpoint.restore(restore_path)
    curr_decoded = model.decompress(arrays)
    row=int(args.input_file.split('/')[-1].split('.')[0])

    # Write reconstructed images out as PNG files.
    for col in range(np.shape(curr_decoded)[1]):
        img = curr_decoded[0,col,:,:,:]/255
        save_img(args.output_file,0,img,row,col+1)


def train(args):
    """Trains the model."""    
    num_epochs = args.num_epochs
    batch_size=args.batch_size
    image_train_path = args.train_path
    num_batches = int(np.ceil(len(image_train_path)/batch_size))
    last_step = int(np.ceil((len(image_train_path)*num_epochs)/batch_size))
    
    with tf.device("/cpu:0"):
        # Create list of sorted list containing image paths within each folder.
        num_cores = multiprocessing.cpu_count()
        image_train_path = Parallel(n_jobs=num_cores)(delayed(make_mat)(image_train_path[i]) for i in range(len(image_train_path)))
        # Create input data pipeline.
        dataset = tf.data.Dataset.from_tensor_slices(image_train_path)
        dataset = dataset.shuffle(buffer_size=len(image_train_path)).repeat(num_epochs)
        dataset = dataset.map(preprocess_data, num_parallel_calls=num_cores)
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(32)

    model = CompressionModel(args)
    step_counter = model.optimizer.iterations

    # Create checkpoint manager and restore the checkpoint if available.
    checkpoint = tf.train.Checkpoint(model=model)
    checkpoint_manager = tf.train.CheckpointManager(
      checkpoint, args.checkpoint_dir, max_to_keep=3,
      step_counter=step_counter, checkpoint_interval=args.checkpoint_interval)
    restore_path = tf.train.latest_checkpoint(args.checkpoint_dir) 
    if restore_path:
        print("Last checkpoint:", restore_path)
        restore_status = checkpoint.restore(restore_path)
        step = step_counter.numpy()
        epochNo=int(np.floor((step+1)/num_batches))
    else:
        restore_status = None
        epochNo=0

    # Perform training.
    for batch_frames,batch_feature1,batch_feature2,batch_feature3,batch_feature4,batch_feature5,batch_feature6,\
    batch_feature7,batch_feature8 in dataset:       
        step = step_counter.numpy()
        curr_loss,curr_bpp,curr_mse = model.train_one_step(batch_frames,batch_feature1,batch_feature2,batch_feature3,
                                        batch_feature4,batch_feature5,batch_feature6,batch_feature7,batch_feature8)
        print("Epoch: %d Step: %d  loss: %f bpp: %f mse: %f\n" % (epochNo,step,curr_loss,curr_bpp,curr_mse))
        if restore_status is not None:
            restore_status.assert_consumed()
            restore_status = None
        if (step+1)%num_batches==0:
            epochNo+=1
        finished = (step + 1 >= last_step)
        checkpoint_manager.save(check_interval=not finished)
        
def parse_args(argv):
	"""Parses command line arguments."""
	parser = argparse_flags.ArgumentParser(
	    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

	# High-level options.
	parser.add_argument(
	    "--checkpoint_dir", default="checkpoints",
	    help="Directory where to save/load model checkpoints.")

	subparsers = parser.add_subparsers(
	    title="commands", dest="command",
	    help="What to do: 'train' loads training data and trains (or continues "
	         "to train) a new model. 'compress' reads an LF image row "
	         "and writes a compressed binary file. 'decompress' "
	         "reads a binary file and reconstructs the images (in PNG format). "
	         "input and output filenames need to be provided for the latter "
	         "two options. Invoke '<command> -h' for more information.")

	# 'train' subcommand.
	train_cmd = subparsers.add_parser(
	    "train",
	    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
	    description="Trains (or continues to train) a new model.")
	train_cmd.add_argument(
	    "--train_path", default="train",
	    help="Path to training data. Data must be in row-wise"
	         "64x64 patches PNG format.")
	train_cmd.add_argument(
	    "--batch_size", type=int, default=30,
	    help="Batch size for training.")
    train_cmd.add_argument(
	    "--lmbda", type=float, default=0.002, dest="lmbda",
	    help="lambda for rate-distortion tradeoff.")
	train_cmd.add_argument(
	    "--num_epochs", type=int, default=100,
	    help="Train up to this number of epochs.")
	train_cmd.add_argument(
	    "--checkpoint_interval", type=int, default=1000,
	    help="Write a checkpoint every `checkpoint_interval` training steps.")

	# 'compress' subcommand.
	compress_cmd = subparsers.add_parser(
	    "compress",
	    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
	    description="Reads a row of LF image, compresses it, and writes a TFCI file.")

	# 'decompress' subcommand.
	decompress_cmd = subparsers.add_parser(
	    "decompress",
	    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
	    description="Reads a TFCI file, reconstructs the images, and writes back "
	                "the PNG files.")

	# Arguments for both 'compress' and 'decompress'.
	for cmd in ((compress_cmd), (decompress_cmd)):
	    cmd.add_argument(
	        "input_file",
	        help="Input filename.")
	    cmd.add_argument(
	        "output_file",
	        help="Output filename.")

	# Parse arguments.
	args = parser.parse_args(argv[1:])
	if args.command is None:
	    parser.print_usage()
	    sys.exit(2)
	return args            

def main(args):
    # Invoke subcommand. 
    if args.command == "train":
	    image_train_path = args.train_path
	    # Read all folder names into a list.
	    image_train_path = make_mat(image_train_path)
	    random.shuffle(image_train_path)
	    args.train_path = image_train_path
	    # Create a checkpoint directory.
	    ckpt_path = args.checkpoint_dir
	    if not os.path.exists(ckpt_path):
	        os.mkdir(ckpt_path)
	    train(args)
    elif args.command == "compress":
	    # Create a directory to store output bitsreams and reconstructions.
	    if not os.path.exists(args.output_file):
	        os.mkdir(args.output_file)
	    compress(args)
  	elif args.command == "decompress":
	    # Create a directory to store output reconstructions.
	    if not os.path.exists(args.output_file):
	        os.mkdir(args.output_file)
	    decompress(args)

      
if __name__ == "__main__":
  app.run(main, flags_parser=parse_args)
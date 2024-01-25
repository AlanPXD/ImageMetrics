from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops.image_ops_impl import _ssim_helper, _fspecial_gauss
from math import log
from tensorflow.keras.metrics import Metric	
from tensorflow._api.v2.image import ssim
import tensorflow as tf
import numpy as np

def _ssim_map_per_channel(img1,
						img2,
						max_val=1.0,
						filter_size=11,
						filter_sigma=1.5,
						k1=0.01,
						k2=0.03,
						keep_padding = True):
	"""Computes SSIM index between img1 and img2 per color channel.

	This function matches the standard SSIM implementation from:
	Wang, Z., Bovik, A. C., Sheikh, H. R., & Simoncelli, E. P. (2004). Image
	quality assessment: from error visibility to structural similarity. IEEE
	transactions on image processing.

	Details:
		- 11x11 Gaussian filter of width 1.5 is used.
		- k1 = 0.01, k2 = 0.03 as in the original paper.

	Args:
		img1: First image batch.
		img2: Second image batch.
		max_val: The dynamic range of the images (i.e., the difference between the
		maximum the and minimum allowed values).
		filter_size: Default value 11 (size of gaussian filter).
		filter_sigma: Default value 1.5 (width of gaussian filter).
		k1: Default value 0.01
		k2: Default value 0.03 (SSIM is less sensitivity to K2 for lower values, so
		it would be better if we took the values in the range of 0 < K2 < 0.4).

	Returns:
		The ssim map for the imgs with a shape like: [..., img_dim_1 - kernel_dim_1 + 1, img_dim_2 - kernel_dim_2 + 1, n_chanels]
	"""
	filter_size = constant_op.constant(filter_size, dtype=dtypes.int32)
	filter_sigma = constant_op.constant(filter_sigma, dtype=img1.dtype)

	shape1, shape2 = array_ops.shape_n([img1, img2])
	checks = [
		control_flow_ops.Assert(
			math_ops.reduce_all(
				math_ops.greater_equal(shape1[-3:-1], filter_size)),
			[shape1, filter_size],
			summarize=8),
		control_flow_ops.Assert(
			math_ops.reduce_all(
				math_ops.greater_equal(shape2[-3:-1], filter_size)),
			[shape2, filter_size],
			summarize=8)
	]

	# Enforce the check to run before computation.
	with ops.control_dependencies(checks):
		img1 = array_ops.identity(img1)

	# TODO(sjhwang): Try to cache kernels and compensation factor.
	kernel = _fspecial_gauss(filter_size, filter_sigma)
	kernel = array_ops.tile(kernel, multiples=[1, 1, shape1[-1], 1])

	# The correct compensation factor is `1.0 - tf.reduce_sum(tf.square(kernel))`,
	# but to match MATLAB implementation of MS-SSIM, we use 1.0 instead.
	compensation = 1.0

	# TODO(sjhwang): Try FFT.
	# TODO(sjhwang): Gaussian kernel is separable in space. Consider applying
	#   1-by-n and n-by-1 Gaussian filters instead of an n-by-n filter.
	def reducer(x):
		shape = array_ops.shape(x)
		x = array_ops.reshape(x, shape=array_ops.concat([[-1], shape[-3:]], 0))
		y = nn.depthwise_conv2d(x, kernel, strides=[1, 1, 1, 1], padding = 'SAME' if keep_padding else 'VALID')
		return array_ops.reshape(
			y, array_ops.concat([shape[:-3], array_ops.shape(y)[1:]], 0))

	luminance, cs = _ssim_helper(img1, img2, reducer, max_val, compensation, k1,
								k2)

	ssim_map = luminance*cs

	return ssim_map

  


def get_weight_map (magnitude_gradient: tf.Tensor,
                    threshold_for_edges,
                    threshold_for_textures,
					weight_for_edges,
                	weight_for_texture,
                	weight_for_smooth):
	"""
		Returns a map of weights that'll be multiplied element-wise with every element of the ssim map
	"""
	max_gradient_magnitude_per_image = tf.reduce_max(magnitude_gradient, axis = (-1, -2 , -3), keepdims=True)

	# removing zeros from the tensor
	max_gradient_magnitude_per_image = tf.where ( max_gradient_magnitude_per_image != 0.0, 
												max_gradient_magnitude_per_image,
												1.0)

	normalized_magnitude_gradient = magnitude_gradient/max_gradient_magnitude_per_image

	#edge_indexes = tf.where( normalized_magnitude_gradient >= threshold_for_edges)
	#texture_indexes = tf.where( tf.math.logical_and( normalized_magnitude_gradient >= threshold_for_textures, normalized_magnitude_gradient < threshold_for_edges) )
	#smooth_indexes = tf.where( normalized_magnitude_gradient < threshold_for_textures)

	# Creating a map with the respective weight for every element of the ssim map
	weight_map = tf.where(normalized_magnitude_gradient >= threshold_for_edges,
		       			  tf.constant(weight_for_edges, dtype = magnitude_gradient.dtype),
						   tf.constant(weight_for_texture, dtype = magnitude_gradient.dtype))
	
	weight_map = tf.where(normalized_magnitude_gradient < threshold_for_textures,
		       			   tf.constant(weight_for_smooth, dtype = magnitude_gradient.dtype),
						  weight_map)
	

	return weight_map

@tf.function
def three_ssim (original_images: tf.Tensor,
                degraded_images: tf.Tensor,
                max_val=255.0,
                weight_for_edges = 3,
                weight_for_texture = 1,
                weight_for_smooth = 1,
				threshold_for_edges = 0.12,
				threshold_for_textures = 0.06,
                filter_size=11,
                filter_sigma=1.5,
                k1=0.01,
                k2=0.03,
                keep_padding = True) -> tf.Tensor:
	"""
		Computes the SSIM (3-SSIM) modified version described in the article "Content-weighted video quality 
		assessment using a three-component image model". This metric allows the ssim in some regions of the image be more 
		significant tham others, in edge regions for example.

		### Warning

		### Parameters
	"""
	ssim_map = _ssim_map_per_channel(original_images, degraded_images, 
									max_val, filter_size, filter_sigma, k1, k2, keep_padding)

	img_grad = tf.image.sobel_edges(original_images)
	imgs_magnitude_grad = tf.sqrt( tf.square(img_grad[:,:,:,:, 0]) + tf.square(img_grad[:,:,:,:, 1]))

	if not keep_padding: # make the magnitude gradient have the same dimensions of the ssim_map
		size_decrease = (filter_size - 1)//2
		imgs_magnitude_grad = imgs_magnitude_grad[:, size_decrease:-size_decrease, size_decrease:-size_decrease]
	
	weight_map = get_weight_map(magnitude_gradient = imgs_magnitude_grad,
								threshold_for_edges = threshold_for_edges,
                    			threshold_for_textures = threshold_for_textures,
								weight_for_edges = weight_for_edges,
                				weight_for_texture = weight_for_texture,
                				weight_for_smooth = weight_for_smooth)
  

	weighted_ssim_sum = tf.reduce_sum(tf.multiply(ssim_map, weight_map), axis = (-1,-2,-3))
	ssim3: tf.Tensor = weighted_ssim_sum/tf.reduce_sum(weight_map, axis = (-1, -2, -3))
  

	return ssim3



def blocking_efect_factor (im: tf.Tensor, block_size = 8) -> tf.Tensor:
	"""

	"""
	n_imgs, height, width, channels = im.shape

	h = np.array(range(0, width - 1))
	h_b = np.array(range(block_size - 1, width - 1, block_size))
	h_bc = np.array(list(set(h).symmetric_difference(h_b)))

	v = np.array(range(0, height - 1))
	v_b = np.array(range(block_size - 1, height - 1, block_size))
	v_bc = np.array(list(set(v).symmetric_difference(v_b)))

	d_b = 0
	d_bc = 0

	# h_b for loop
	for i in list(h_b):
		diff = im[:, :, i] - im[:, :, i+1]
		d_b += tf.reduce_sum(tf.math.square(diff), axis = [1,2])

	# h_bc for loop
	for i in list(h_bc):
		diff = im[:, :, i] - im[:, :, i+1]
		d_bc += tf.reduce_sum(tf.math.square(diff), axis = [1,2])

	# v_b for loop
	for j in list(v_b):
		diff = im[:, j, :] - im[:, j+1, :]
		d_b += tf.reduce_sum(tf.math.square(diff), axis = [1,2])

	# V_bc for loop
	for j in list(v_bc):
		diff = im[:, j, :] - im[:, j+1, :]
		d_bc += tf.reduce_sum(tf.math.square(diff), axis = [1,2])

	# N code
	n_hb = height * (width/block_size) - 1
	n_hbc = (height * (width - 1)) - n_hb
	n_vb = width * (height/block_size) - 1
	n_vbc = (width * (height - 1)) - n_vb

	# D code
	d_b /= (n_hb + n_vb)
	d_bc /= (n_hbc + n_vbc)

	# Log
	t = np.log2(block_size)/np.log2(min(height, width))
	
	# BEF
	bef = t*(d_b - d_bc)

	return tf.nn.relu(bef)



def psnrb (target_imgs: tf.Tensor, degraded_imgs: tf.Tensor, ) -> tf.Tensor:
	"""
	Computes the PSNR-B for a batch of images

	### Obs: 
	The order of images are important, the block efect factor (BEF) calculation only depends of one image.

	### Ref:
	Quality Assessment of Deblocked Images: Changhoon Yim, Member, IEEE, and Alan Conrad Bovik, Fellow, IEEE 
	"""
	imgs_shape = degraded_imgs.shape.__len__()

	assert imgs_shape == 4

	img_mse = tf.reduce_mean(tf.square(target_imgs - degraded_imgs), axis=[1,2,3])

	bef_total = blocking_efect_factor(degraded_imgs)

	psnr_b =  tf.math.add(-10*tf.math.log(bef_total + img_mse)/log(10), 10*log(255**2, 10))

	return psnr_b

class PSNRB (Metric):
    
	def __init__(
    self,
    name='psnrb',
    **kwargs):
		super().__init__(name=name, **kwargs)
	

	def update_state(self, y_true, y_pred, sample_weight=None):
		self.result = psnrb(y_true, y_pred)

	def result(self):
		return self.result

	def __call__(self, y_true, y_pred, sample_weight=None):
		return psnrb(y_true, y_pred)


class SSIM (Metric):
    
	def __init__(
		self,
		name='ssim',
		max_val = 1,
		filter_size: int = 11,
		filter_sigma: float = 1.5,
		k1: float = 0.01,
		k2: float = 0.03,
    **kwargs):
		
		super().__init__(name=name, **kwargs)
		self.max_val = max_val
		self.filter_size = filter_size
		self.filter_sigma = filter_sigma
		self.k1 = k1
		self.k2 = k2
		
	def update_state(self, y_true, y_pred, sample_weight=None):
		self.result = ssim(y_true, y_pred, self.max_val,
                        self.filter_size,
                        self.filter_sigma,
                        self.k1,
                        self.k2)

	def result(self):
		return self.result

	def __call__(self, y_true, y_pred, sample_weight=None):
		return ssim(y_true, y_pred, self.max_val,
						self.filter_size,
						self.filter_sigma,
						self.k1,
						self.k2)

class SSIM3 (Metric):
    
	def __init__(
		self,
		name='3ssim',
		max_val = 1,
  		weight_for_edges: int = 3,
        weight_for_texture: int = 1,
        weight_for_smooth: int = 1,
        threshold_for_edges: float = 0.12,
        threshold_for_textures: float = 0.06,
		filter_size: int = 11,
		filter_sigma: float = 1.5,
		k1: float = 0.01,
		k2: float = 0.03,
    **kwargs):
		
		super().__init__(name=name, **kwargs)
		self.max_val = max_val
		self.weight_for_edges: int = weight_for_edges,
		self.weight_for_texture: int = weight_for_texture,
		self.weight_for_smooth: int = weight_for_smooth,
		self.threshold_for_edges: float = threshold_for_edges,
		self.threshold_for_textures: float = threshold_for_textures,
		self.filter_size = filter_size
		self.filter_sigma = filter_sigma
		self.k1 = k1
		self.k2 = k2
		
	def update_state(self, y_true, y_pred, sample_weight=None):
		self.result = three_ssim(y_true, y_pred, self.max_val,
                        self.weight_for_edges,
                        self.weight_for_texture,
                        self.weight_for_smooth,
                        self.threshold_for_edges,
                        self.threshold_for_textures,
                        self.filter_size,
                        self.filter_sigma,
                        self.k1,
                        self.k2)

	def result(self):
		return self.result

	def __call__(self, y_true, y_pred, sample_weight=None):
		return three_ssim(y_true, y_pred, self.max_val,
                        self.weight_for_edges,
                        self.weight_for_texture,
                        self.weight_for_smooth,
                        self.threshold_for_edges,
                        self.threshold_for_textures,
                        self.filter_size,
                        self.filter_sigma,
                        self.k1,
                        self.k2)

if __name__ == "__main__":
	
	from os import environ
	environ["CUDA_VISIBLE_DEVICES"]="2"
	
	img1_np = 4* np.ones(shape=(2,12,12,1))#np.random.normal(127, 25, size = (2,64,64,1))
	img2_np = 4 * np.ones(shape=(2,12,12,1))#np.random.normal(127, 35/3, size = (2,64,64,1))
	img1 = tf.constant(img1_np, dtype="float64")
	img2 = tf.constant(img2_np, dtype="float64")

	print(three_ssim(img1, img2, 255, weight_for_edges=2., weight_for_smooth= 1., weight_for_texture = 1., keep_padding=False))
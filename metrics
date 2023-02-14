from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops.image_ops_impl import _ssim_helper, _fspecial_gauss
from math import log
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

def get_regions_indexes (magnitude_gradient: tf.Tensor,
                         threshold_for_edges: float = 0.12,
                         threshold_for_texture: float = 0.06):
  """
  """
  max_gradient_magnitude_per_image = tf.reduce_max(magnitude_gradient, axis = (-1, -2 , -3), keepdims=True)
  normalized_magnitude_gradient = magnitude_gradient/max_gradient_magnitude_per_image

  edge_indexes = tf.where( normalized_magnitude_gradient >= threshold_for_edges)
  texture_indexes = tf.where( tf.math.logical_and( normalized_magnitude_gradient >= threshold_for_texture, normalized_magnitude_gradient < threshold_for_edges) )
  smooth_indexes = tf.where( normalized_magnitude_gradient < threshold_for_texture)

  return edge_indexes, texture_indexes, smooth_indexes
    
def three_ssim (original_images,
                degraded_images,
                max_val=1.0,
                weight_for_edges = 2,
                weight_for_texture = 1,
                weight_for_smooth = 1,
                filter_size=11,
                filter_sigma=1.5,
                k1=0.01,
                k2=0.03,
                keep_padding = True):
  """
    Computes the SSIM (3-SSIM) modified version described in the article "Content-weighted video quality 
    assessment using a three-component image model". This metric allows the ssim in some regions of the image be more 
    significant tham others, in edge regions for example.

    ### Warning

    ### Parameters
  """
  ssim_map = _ssim_map_per_channel(original_images, degraded_images, max_val, filter_size, filter_sigma, k1, k2, keep_padding)

  img_grad = tf.image.sobel_edges(original_images)
  imgs_magnitude_grad = tf.sqrt( tf.square(img_grad[:,:,:,:, 0]) + tf.square(img_grad[:,:,:,:, 1]))

  if not keep_padding: # make the magnitude gradient have the same dimensions of the ssim_map
    size_decrease = (filter_size - 1)//2
    imgs_magnitude_grad = imgs_magnitude_grad[:, size_decrease:-size_decrease, size_decrease:-size_decrease]
    print(math_ops.reduce_mean(ssim_map, axis = (-1, -2, -3)))
  
  edge_indexes, texture_indexes, smooth_indexes = get_regions_indexes (imgs_magnitude_grad, threshold_for_edges = 0.12, threshold_for_texture = 0.06)

  zero_map = tf.zeros(shape=imgs_magnitude_grad.shape , dtype = tf.float64)

  # Calculate the ssim_map with zeros in nom edege/texture/smooth regions.

  edge_mask = tf.tensor_scatter_nd_update(zero_map, edge_indexes, tf.constant(1., shape = edge_indexes.shape[0], dtype=tf.float64))
  ssim_map_for_edges = tf.math.multiply(ssim_map, edge_mask)

  texture_mask = tf.tensor_scatter_nd_update(zero_map, texture_indexes, tf.constant(1., shape = texture_indexes.shape[0], dtype=tf.float64))
  ssim_map_for_texture = tf.math.multiply(ssim_map, texture_mask)

  smooth_mask = tf.tensor_scatter_nd_update(zero_map, smooth_indexes, tf.constant(1., shape = smooth_indexes.shape[0], dtype=tf.float64))
  ssim_map_for_smooth = tf.math.multiply(ssim_map, smooth_mask)

  ones = edge_mask + texture_mask + smooth_mask

  # Calculate the number of nom zero elements per mask

  n_edge_pixels_per_image = tf.reduce_sum(edge_mask, axis = (-1, -2, -3))
  n_texture_pixels_per_image = tf.reduce_sum(texture_mask, axis = (-1, -2, -3))
  n_smooth_pixels_per_image = tf.reduce_sum(smooth_mask, axis = (-1, -2, -3))

  # 3 component calculations

  ssim_on_edges = tf.reduce_sum(ssim_map_for_edges, axis = (-1 ,-2, -3))/n_edge_pixels_per_image
  ssim_on_textures = tf.reduce_sum(ssim_map_for_texture, axis = (-1 ,-2, -3))/n_texture_pixels_per_image
  ssim_on_smooth = tf.reduce_sum(ssim_map_for_smooth, axis = (-1 ,-2, -3))/n_smooth_pixels_per_image
  
  ssim3 = ( weight_for_edges*n_edge_pixels_per_image*ssim_on_edges + weight_for_texture*n_texture_pixels_per_image*ssim_on_textures + 
           weight_for_smooth*n_smooth_pixels_per_image*ssim_on_smooth ) / (weight_for_edges*n_edge_pixels_per_image 
          + weight_for_texture*n_texture_pixels_per_image + weight_for_smooth*n_smooth_pixels_per_image)
  
  return ssim3


@tf.function
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

	return tf.math.maximum(bef, tf.zeros(bef.shape, dtype = bef.dtype))



def psnrb (target_imgs: tf.Tensor, degraded_imgs: tf.Tensor, ) -> tf.float32:
	"""
	Computes the PSNR-B for a batch of images

	### Obs: 
	The order of images are important, 'couse of the block efect factor (BEF) calculation depends only of one image.

	### Ref:
	Quality Assessment of Deblocked Images: Changhoon Yim, Member, IEEE, and Alan Conrad Bovik, Fellow, IEEE 
	"""
	imgs_shape = degraded_imgs.shape.__len__()

	assert imgs_shape == 4

	img_mse = tf.reduce_mean(tf.square(target_imgs - degraded_imgs), axis=[1,2,3])

	bef_total = blocking_efect_factor(degraded_imgs)

	psnr_b =  tf.math.add(-10*tf.math.log(bef_total + img_mse)/log(10), 10*log(255**2, 10))

	return psnr_b

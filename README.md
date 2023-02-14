# ImageMetrics

This repository has implementations of some image metrics made with TensorFlow. 

- 3-SSIM 

    This function is a modification of SSIM (Structural similarity index measure), described in the following article:

[1] A. C. Bovik, “Content-weighted video quality assessment using a three-component image model”, J. Electron. Imaging, vol. 19, nº 1, p. 011003, jan. 2010, doi: 10.1117/1.3267087.


- PSNRB

- Obs: This function is an adapted version of the PSNR-B implemented in SEWAR lib, modified for operating over a batch of images. The original function is implemented with Numpy, which made it incompatible with the graph execution. Furthermore, the function was made to operate over 2 images, instead of two batches of images. \\

This Metric is also a modified version of the PSNR, which was a famous metric, in that case, applied for images. The modification consists of an addition of a measure of blocking effects, caused by compression. Details are described in the article:

[1] Changhoon Yim e A. C. Bovik, “Quality Assessment of Deblocked Images”, IEEE Trans. on Image Process., vol. 20, nº 1, p. 88–98, jan. 2011, doi: 10.1109/TIP.2010.2061859.

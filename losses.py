from typing import Any

from tensorflow import Tensor

from tensorflow.keras.losses import Reduction, Loss
from tensorflow.python.keras.utils.losses_utils import ReductionV2

from tensorflow._api.v2.image import ssim
from ImageMetrics.metrics import three_ssim, psnrb

class LSSIM(Loss):
    
    def __init__(
        self,
        max_val:Any = 1,
        filter_size: int = 11,
        filter_sigma: float = 1.5,
        k1: float = 0.01,
        k2: float = 0.03,
        reduction = Reduction.AUTO,
        name = "LSSIM"
    ) -> None:
        super().__init__(reduction, name)    
        self.max_val = max_val
        self.filter_size = filter_size
        self.filter_sigma = filter_sigma
        self.k1 = k1
        self.k2 = k2
        self.reduction = reduction
    
    def __call__(self, y_true, y_pred, sample_weight = None) -> Tensor:
        return 1 - ssim(y_true, y_pred, self.max_val,
                        self.filter_size,
                        self.filter_sigma,
                        self.k1,
                        self.k2)

class L3SSIM(Loss):
    
    def __init__(
        self,
        max_val:Any = 1,
        weight_for_edges: int = 3,
        weight_for_texture: int = 1,
        weight_for_smooth: int = 1,
        threshold_for_edges: float = 0.12,
        threshold_for_textures: float = 0.06,
        filter_size: int = 11,
        filter_sigma: float = 1.5,
        k1: float = 0.01,
        k2: float = 0.03,
        keep_padding = True,
        reduction = Reduction.AUTO,
        name = "L3SSIM"
    ) -> None:
        super().__init__(reduction, name)    
        self.max_val = max_val
        self.weight_for_edges: int = weight_for_edges,
        self.weight_for_texture: int = weight_for_texture,
        self.weight_for_smooth: int = weight_for_smooth,
        self.threshold_for_edges: float = threshold_for_edges,
        self.threshold_for_textures: float = threshold_for_textures,
        self.filter_size = filter_size,
        self.filter_sigma = filter_sigma,
        self.k1 = k1,
        self.k2 = k2,
        self.keep_padding = keep_padding
        self.reduction = reduction
    
    def __call__(self, y_true, y_pred, sample_weight = None) -> Tensor:
        
        return 1 - three_ssim(y_true, y_pred, self.max_val,
                        self.weight_for_edges,
                        self.weight_for_texture,
                        self.weight_for_smooth,
                        self.threshold_for_edges,
                        self.threshold_for_textures,
                        self.filter_size,
                        self.filter_sigma,
                        self.k1,
                        self.k2)
class LPSNRB(Loss):
    
    def __init__(self, reduction=Reduction.AUTO, name="LPSNRB"):
        super().__init__(reduction, name)    
    
    def __call__(self, y_true, y_pred, sample_weight = None) -> Tensor:
        
        return -psnrb(target_imgs=y_true, degraded_imgs=y_pred)
    
    
    

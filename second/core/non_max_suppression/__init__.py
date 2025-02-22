from second.core.non_max_suppression.nms_cpu import nms_jit, soft_nms_jit
import torch

if torch.cuda.is_available():
  from second.core.non_max_suppression.nms_gpu import (nms_gpu, rotate_iou_gpu,
                                                        rotate_nms_gpu)

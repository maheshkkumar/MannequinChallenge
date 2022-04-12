import imageio as io
import numpy as np
import torch
from skimage.transform import resize
from torch.autograd import Variable

from .config import _C as cfg
from .models import pix2pix_model

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

class MannequinChallenge():
    def __init__(self):
        self.eval_num_threads = 2
        self.opt = cfg
        self.model = pix2pix_model.Pix2PixModel(opt=self.opt, _isTrain=False)
        
    def evaluate(self, img):
        img = np.float32(io.imread(img))/255.0
        original_height, original_width, _ = img.shape
        new_height, new_width = 384, 512
        img = resize(img, (new_height, new_width), order = 1)
        input_img =  torch.from_numpy( np.transpose(img, (2,0,1)) ).contiguous().float()
        input_img = input_img.unsqueeze(0)

        input_images = Variable(input_img.to(device) )
        pred_log_depth, _ = self.model.netG.forward(input_images)
        pred_log_depth = torch.squeeze(pred_log_depth)
        pred_depth = torch.exp(pred_log_depth)

        # convert to disparity
        pred_inv_depth = 1.0 / pred_depth
        pred_inv_depth = pred_inv_depth.data.cpu().numpy()

        # min-max normalization
        pred_inv_depth = (pred_inv_depth - np.min(pred_inv_depth)) / (np.max(pred_inv_depth) - np.min(pred_inv_depth))

        # resize to original size
        pred_inv_depth = resize(pred_inv_depth, (original_height, original_width), order = 1)

        return pred_inv_depth 


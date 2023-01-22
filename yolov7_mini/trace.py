import torch
import torchvision
import numpy as np
import torch.nn as nn
from .ensemble import BatchNormXd

class TracedModel(nn.Module):
    """
    Traced Model using Torch Just in Time Compiler.
    """
    def __init__(self, model, device=None, img_size=(640,640)): 
        super(TracedModel, self).__init__()
        
        print(" 🗿 Optimizing model using JIT ") 
        self.stride = model.stride
        self.names  = model.names
        self.model  = model

        self.model = self.revert_sync_batchnorm(self.model)
        self.model.to('cpu')
        self.model.eval()

        self.detect_layer = self.model.model[-1]
        self.model.traced = True
        
        rand_example = torch.rand(1, 3, *img_size)
        
        traced_script_module = torch.jit.trace(self.model, rand_example, strict=False)
        #traced_script_module = torch.jit.script(self.model)

        self.model = traced_script_module
        self.model.to(device)
        self.detect_layer.to(device)
        print(" 🛸 Traced Model Conversion successful! ") 

    def forward(self, x, augment=False, profile=False):
        """
        Forward pass for Traced Model.
        """
        out = self.model(x)
        out = self.detect_layer(out)
        return out

    
    def revert_sync_batchnorm(self, module):
        # this is very similar to the function that it is trying to revert:
        # https://github.com/pytorch/pytorch/blob/c8b3686a3e4ba63dc59e5dcfe5db3430df256833/torch/nn/modules/batchnorm.py#L679
        module_output = module
        if isinstance(module, torch.nn.modules.batchnorm.SyncBatchNorm):
            new_cls = BatchNormXd
            module_output = BatchNormXd(module.num_features,
                                                module.eps, module.momentum,
                                                module.affine,
                                                module.track_running_stats)
            if module.affine:
                with torch.no_grad():
                    module_output.weight = module.weight
                    module_output.bias = module.bias
            module_output.running_mean = module.running_mean
            module_output.running_var = module.running_var
            module_output.num_batches_tracked = module.num_batches_tracked
            if hasattr(module, "qconfig"):
                module_output.qconfig = module.qconfig
        for name, child in module.named_children():
            module_output.add_module(name, self.revert_sync_batchnorm(child))
        del module
        return module_output
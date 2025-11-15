import numpy as np
import torch
import cv2

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.model.eval()
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        self._register_hooks()

    def _register_hooks(self):
        def fw_hook(module, inp, out):
            self.activations = out.detach()
        def bw_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()
        self.target_layer.register_forward_hook(fw_hook)
        self.target_layer.register_backward_hook(bw_hook)

    def __call__(self, input_tensor, class_idx: int):
        input_tensor = input_tensor.clone().detach().requires_grad_(True)
        logits = self.model(input_tensor)
        score = logits[0, class_idx]
        self.model.zero_grad()
        score.backward(retain_graph=True)
        weights = self.gradients.mean(dim=(2,3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = torch.relu(cam)
        cam = cam[0,0].cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() + 1e-6)
        H, W = input_tensor.shape[-2], input_tensor.shape[-1]
        cam = cv2.resize(cam, (W, H), interpolation=cv2.INTER_CUBIC)
        return cam
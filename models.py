import os
import torch
import torchxrayvision as xrv

_DEVICE = os.environ.get("DEVICE", "cpu")
_MODEL_NAME = os.environ.get("MODEL_NAME", "densenet121-res224-all")

class ModelSingleton:
    _instance = None
    model = None
    labels = None
    device = _DEVICE

    @classmethod
    def instance(cls):
        if cls._instance is None:
            cls._instance = cls()
            cls._instance._load()
        return cls._instance

    def _load(self):
        # Load model from torchxrayvision
        self.model = xrv.models.get_model(self._safe_name(_MODEL_NAME))
        self.model.eval()
        self.model.to(self.device)

        try:
            self.labels = self.model.labels
        except AttributeError:
            # Fallback to NIH dataset labels (commonly used with pretrained weights)
            print("Model has no `.labels` attribute; using NIH14 label set instead.")
            self.labels = xrv.datasets.default_pathologies
        # --- FIX END ---

    def _safe_name(self, name):
        return name

    @torch.inference_mode()
    def predict(self, img_tensor_np):
        x = torch.from_numpy(img_tensor_np).float().unsqueeze(0)
        x = x.to(self.device)
        out = self.model(x)
        probs = torch.sigmoid(out).detach().cpu().numpy()[0]
        return probs, out


def get_last_conv_layer(model: torch.nn.Module):
    last = None
    for m in model.modules():
        if isinstance(m, torch.nn.Conv2d):
            last = m
    return last
import io
import numpy as np
from PIL import Image
from skimage.transform import resize
import pydicom
from pydicom.pixel_data_handlers.util import apply_voi_lut
import torchxrayvision as xrv

ALLOWED_EXTS = {'.jpg', '.jpeg', '.png', '.dcm'}
MAX_BYTES = 10 * 1024 * 1024  # 10 MB

def is_allowed(filename: str) -> bool:
    return any(filename.lower().endswith(ext) for ext in ALLOWED_EXTS)

def load_image_bytes_safe(file_storage) -> bytes:
    file_storage.stream.seek(0, 2)
    size = file_storage.stream.tell()
    file_storage.stream.seek(0)
    if size > MAX_BYTES:
        raise ValueError("File too large. Max 10 MB.")
    return file_storage.read()

def read_dicom_deidentify(data: bytes) -> Image.Image:
    ds = pydicom.dcmread(io.BytesIO(data), force=True)
    ds.remove_private_tags()
    for tag in [
        (0x0010,0x0010),(0x0010,0x0020),(0x0010,0x0030),(0x0010,0x0040),
        (0x0008,0x0090),(0x0008,0x0080),(0x0008,0x0081),
        (0x0010,0x1000),(0x0010,0x1001),(0x0010,0x2160),(0x0010,0x4000)
    ]:
        if tag in ds:
            del ds[tag]
    arr = ds.pixel_array
    if hasattr(ds, "PhotometricInterpretation") and ds.PhotometricInterpretation == "MONOCHROME1":
        arr = np.max(arr) - arr
    try:
        arr = apply_voi_lut(arr, ds)
    except Exception:
        pass
    arr = arr.astype(np.float32)
    arr -= arr.min()
    if arr.max() > 0:
        arr /= arr.max()
    arr = (arr * 255.0).clip(0,255)
    im = Image.fromarray(arr.astype(np.uint8), mode="L")
    return im

def read_image(data: bytes, filename: str) -> Image.Image:
    if filename.lower().endswith(".dcm"):
        return read_dicom_deidentify(data)
    im = Image.open(io.BytesIO(data))
    if im.mode != "L":
        im = im.convert("L")
    return im

def to_model_tensor(pil_im: Image.Image, target_size=224):
    img = np.array(pil_im, dtype=np.float32)
    img_resized = resize(img, (target_size, target_size), preserve_range=True, anti_aliasing=True).astype(np.float32)
    img_resized = img_resized[None, ...]
    img_norm = xrv.datasets.normalize(img_resized, maxval=255)
    return img_norm
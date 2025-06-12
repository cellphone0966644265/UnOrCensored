
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from collections import OrderedDict

def imread(path, mod='cv2', loadsize=None, rgb=False):
    if mod == 'cv2':
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is None: raise IOError(f"Không thể đọc file ảnh: {path}")
        if len(img.shape) == 2: img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        if img.shape[2] == 4: img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        if rgb: img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    elif mod == 'gray':
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if loadsize is not None:
        img = cv2.resize(img, (loadsize, loadsize))
    return img

def imwrite(path, img):
    cv2.imwrite(path, img)

def im2tensor(image_numpy, bgr2rgb=True, is0_1=False, gpu_id='-1'):
    if bgr2rgb:
        image_numpy = cv2.cvtColor(image_numpy, cv2.COLOR_BGR2RGB)
    img_tensor = torch.from_numpy(image_numpy.transpose((2, 0, 1))).float()
    if not is0_1:
        img_tensor = (img_tensor / 255.0 - 0.5) / 0.5
    else:
        img_tensor = img_tensor / 255.0
    img_tensor = img_tensor.unsqueeze(0)
    if gpu_id != '-1' and torch.cuda.is_available():
        img_tensor = img_tensor.cuda()
    return img_tensor

def tensor2im(tensor):
    tensor = tensor.squeeze(0).cpu().float().detach().numpy()
    tensor = ((tensor + 1) / 2.0 * 255.0).clip(0, 255)
    tensor = tensor.transpose((1, 2, 0))
    if tensor.shape[2] == 3:
        tensor = cv2.cvtColor(tensor, cv2.COLOR_RGB2BGR)
    return tensor.astype(np.uint8)

def run_segment(img, net, gpu_id='-1', size=360):
    img_size = img.shape
    img_resized = cv2.resize(img, (size, size), interpolation=cv2.INTER_CUBIC)
    img_tensor = im2tensor(img_resized, bgr2rgb=False, is0_1=True, gpu_id=gpu_id)
    with torch.no_grad():
        mask_tensor = net(img_tensor)
        if isinstance(mask_tensor, tuple):
            mask_tensor = mask_tensor[0]
    mask = mask_tensor.squeeze().cpu().numpy()
    mask = cv2.resize(mask, (img_size[1], img_size[0]))
    return (mask * 255).astype(np.uint8)

def mask_threshold(mask, threshold, ex_mun):
    _, mask_bin = cv2.threshold(mask, threshold, 255, cv2.THRESH_BINARY)
    if ex_mun > 0:
        kernel = np.ones((ex_mun, ex_mun), np.uint8)
        mask_bin = cv2.dilate(mask_bin, kernel, iterations=1)
    return mask_bin

def find_mostlikely_ROI(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours: return mask
    max_contour = max(contours, key=cv2.contourArea)
    output_mask = np.zeros_like(mask)
    cv2.drawContours(output_mask, [max_contour], -1, 255, thickness=cv2.FILLED)
    return output_mask

def addmosaic_base(img, mask, n, mod):
    img_m = img.copy()
    h, w = img.shape[:2]
    if n == 0:
        area = np.count_nonzero(mask)
        n = max(int(np.sqrt(area) / 15), 5)
    for i in range(0, h - n, n):
        for j in range(0, w - n, n):
            if mask[i + n // 2, j + n // 2] > 128:
                color = img[i:i + n, j:j + n].mean(axis=(0, 1))
                img_m[i:i + n, j:j + n] = color
    return img_m

def boundingSquare(mask, Ex_mul):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours: return 0, 0, 0, 0
    max_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(max_contour)
    center_x, center_y = x + w // 2, y + h // 2
    size = int(max(w, h) * Ex_mul)
    half_size = size // 2
    return center_x, center_y, half_size, cv2.contourArea(max_contour)

def run_pix2pix_generator(img, net, gpu_id='-1'):
    img_resized = cv2.resize(img, (256, 256))
    img_tensor = im2tensor(img_resized, bgr2rgb=True, is0_1=False, gpu_id=gpu_id)
    with torch.no_grad():
        fake_img_tensor = net(img_tensor)
    fake_img = tensor2im(fake_img_tensor)
    return cv2.resize(fake_img, (img.shape[1], img.shape[0]))

def crop_and_resize_for_model(img, x, y, size, model_input_size=256):
    """
    Cắt vùng ROI từ ảnh gốc và resize về kích thước cho model.
    Trả về ảnh đã crop và tọa độ gốc để ghép lại.
    """
    y_start, y_end = max(0, y - size), min(img.shape[0], y + size)
    x_start, x_end = max(0, x - size), min(img.shape[1], x + size)
    
    if (y_end - y_start) <= 0 or (x_end - x_start) <= 0:
        return None, None

    img_crop = img[y_start:y_end, x_start:x_end]
    img_crop_resized = cv2.resize(img_crop, (model_input_size, model_input_size), interpolation=cv2.INTER_AREA)
    
    return img_crop_resized, (x_start, y_start, x_end, y_end)

def replace_mosaic(img, fake_img, mask, coords, no_feather=False):
    """
    SỬA ĐỔI: Thay thế vùng ROI bằng ảnh đã xử lý, đảm bảo kích thước khớp.
    'coords' là tuple (x_start, y_start, x_end, y_end) của vùng cần thay thế.
    """
    img_result = img.copy()
    x_start, y_start, x_end, y_end = coords
    
    fake_img_resized = cv2.resize(fake_img, (x_end - x_start, y_end - y_start), interpolation=cv2.INTER_LANCZOS4)

    roi_original = img[y_start:y_end, x_start:x_end]
    
    alpha_mask_full = mask[y_start:y_end, x_start:x_end]
    if no_feather:
        alpha_mask_blurred = cv2.threshold(alpha_mask_full, 128, 255, cv2.THRESH_BINARY)[1]
    else:
        alpha_mask_blurred = cv2.GaussianBlur(alpha_mask_full, (21, 21), 0)

    alpha = np.expand_dims(alpha_mask_blurred.astype(float) / 255.0, axis=2)

    blended = roi_original * (1 - alpha) + fake_img_resized * alpha
    img_result[y_start:y_end, x_start:x_end] = blended.astype(np.uint8)
    
    return img_result

def run_bvdnet_generator(sequence, previous_frame_tensor, net, gpu_id='-1'):
    norm_sequence = [(s.astype(np.float32) / 127.5) - 1.0 for s in sequence]
    seq_np = np.array(norm_sequence)
    
    seq_np_transposed = seq_np.transpose(3, 0, 1, 2) 
    
    seq_torch = torch.from_numpy(seq_np_transposed).unsqueeze(0).float()
    
    if gpu_id != '-1' and torch.cuda.is_available():
        seq_torch = seq_torch.cuda()

    with torch.no_grad():
        generated_frame_tensor = net(seq_torch, previous_frame_tensor)
    
    return tensor2im(generated_frame_tensor), generated_frame_tensor

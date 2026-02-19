import numpy as np
import cv2
import torch


def get_GaborFilters(ksize=15, sigma=4.0, lambd=10.0, gamma=0.5, psi=0, device='cpu', dtype=torch.float32):
    filters = []
    if dtype == torch.float16:
        cv_type = cv2.CV_16F
    elif dtype == torch.float32:
        cv_type = cv2.CV_32F
    elif dtype == torch.float64:
        cv_type = cv2.CV_64F
    else:
        raise ValueError("Unsupported dtype. Supported dtypes are: torch.float16, torch.float32, torch.float64.")
    for theta in np.arange(0, np.pi, np.pi / 4):
        kern = cv2.getGaborKernel((ksize, ksize), sigma, theta, lambd, gamma, psi, ktype=cv_type)
        kern /= kern.sum()
        kern = torch.from_numpy(kern).to(device)
        filters.append(kern)
    return filters
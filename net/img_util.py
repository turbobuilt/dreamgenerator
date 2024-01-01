import torch

def rgb_to_hsl(image):
    # Assuming `image` is a PyTorch tensor with shape [batch_size, 3, height, width]
    # and its values are in the range [0, 1]
    
    epsilon = 1e-6

    # Separate the RGB channels
    R = image[:, 0, :, :]
    G = image[:, 1, :, :]
    B = image[:, 2, :, :]

    # Compute max and min of the components
    max_rgb, _ = image.max(dim=1)
    min_rgb, _ = image.min(dim=1)

    # Lightness
    L = (max_rgb + min_rgb) / 2

    # Calculate Delta
    delta = max_rgb - min_rgb + epsilon
    
    # Saturation
    S = torch.where(L < 0.5, delta / (max_rgb + min_rgb + epsilon), delta / (2 - max_rgb - min_rgb + epsilon))

    # Avoid division by zero for delta == 0
    S = torch.where(delta == 0, torch.zeros_like(S), S)

    # Hue
    R_eq = torch.where(max_rgb == R, ((G - B) / delta) % 6, torch.zeros_like(R))
    G_eq = torch.where(max_rgb == G, ((B - R) / delta) + 2, torch.zeros_like(R))
    B_eq = torch.where(max_rgb == B, ((R - G) / delta) + 4, torch.zeros_like(R))
    H = 60 * (R_eq + G_eq + B_eq)
    H = H / 360.0 # Normalize to the range [0, 1]

    hsl_image = torch.stack([H, S, L], dim=1)
    
    return hsl_image
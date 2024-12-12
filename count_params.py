import torch
import torch.nn as nn
from unet.unet_stvit import UNet_STA
def count_parameters(model):
    """
    Count the total number of parameters in a PyTorch model.
    Args:
        model (torch.nn.Module): The model to analyze.
    Returns:
        dict: A dictionary containing counts of trainable and non-trainable parameters.
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable_params = total_params - trainable_params

    return {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "non_trainable_params": non_trainable_params,
    }

# Example usage with a simple model
if __name__ == "__main__":
    
    # model = VMUNet(input_channels=4, dims=[64, 128, 256, 512], dims_decoder=[512, 256, 128, 64], depths=[3, 4, 5, 6], depths_decoder=[6, 5, 4, 3])
    # model = GGCNN(input_channels=4)
    # model = UNetENN(4)
    model = UNet_STA(1, 9)
    param_counts = count_parameters(model)

    print(f"Total Parameters: {param_counts['total_params']}")
    print(f"Trainable Parameters: {param_counts['trainable_params']}")
    print(f"Non-trainable Parameters: {param_counts['non_trainable_params']}")

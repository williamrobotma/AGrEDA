"""Model utility functions."""

def set_requires_grad(model, requires_grad=True):
    """Sets requires_grad for all parameters in model."""
    for param in model.parameters():
        param.requires_grad = requires_grad

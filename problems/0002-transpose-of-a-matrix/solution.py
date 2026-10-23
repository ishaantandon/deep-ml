import torch

def transpose_matrix(a) -> torch.Tensor:
    """
    Transpose a 2D matrix using PyTorch.
    
    Args:
        a: A 2D matrix (can be list, numpy array, or torch.Tensor)
    
    Returns:
        A transposed torch.Tensor
    """
    a_t = torch.as_tensor(a)
    # Your code here
    rows, cols = a_t.shape
    results = torch.empty((cols, rows), dtype=a_t.dtype)
    for i in range(rows):
        for j in range(cols):
            results[j,i] = a_t[i,j]
    return results
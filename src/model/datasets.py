"""Datasets for models."""
import torch

class SpotDataset(torch.utils.data.Dataset):
    """Dataset for cell spots. Indexes a spot with GEx data, and optionally cell
    type proportion.

    Args:
        X (:obj:, array_like of `float`): An array of normalized log gene
            expression values.
        Y (:obj:, array_like of `float`, optional): An array of cell type
            proportion. Default: ``None``.

    Shape:
        - X: :math: `(N_{spots}, C_{genes})`, where :math: `N_{spots}` is the
        number of spots, and :math: `C_{genes}` is the number of genes.
        - Y: :math: `(N_{spots}, C_{types})`, where :math: `N_{spots}` is the
        number of spots, and :math: `C_{types}` is the number of cell types.

    """
    def __init__(self, X, Y=None):
        super().__init__()

        self.X = torch.as_tensor(X).float()
        assert self.X.dim() == 2, f"X should be rank 2, got {self.X.dim()}"

        if Y is None:
            self.Y = torch.empty((self.X.shape[0], 0), dtype=torch.float)
        else:
            self.Y = torch.as_tensor(Y).float()
            assert self.Y.dim() == 2, f"Y should be rank 2, got {self.Y.dim()}"
            assert len(self.X) == len(self.Y), "X and Y unequal lengths"

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

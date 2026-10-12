""" This module defines the pose read-out head. """
import roma
import torch

# Regressed per edge, a 9D rotation, a 3D translation and two log-variances.
ROT_DIM, TRANS_DIM, CONF_DIM = 9, 3, 2
PRED_WIDTH = ROT_DIM + TRANS_DIM + CONF_DIM


class PoseHead(torch.nn.Module):
    """ Turns per-frame pooled patch descriptors into relative poses and confidences.
        The 9 rotation values are mapped to the nearest rotation in Frobenius norm by
        SVD (Levinson et al., NeurIPS 2020). The tokens arrive patch major, so the
        frame axis splits out as (-1, T) and the mean runs over the patches.
    """

    def __init__(self, embed_dim: int, prediction_dimension: int):
        super().__init__()
        self.prediction_dimension = prediction_dimension
        self.num_frames = prediction_dimension + 1

        # One descriptor per frame, plus the camera token.
        self.layers = torch.nn.Linear(
            embed_dim * (self.num_frames + 1), prediction_dimension * PRED_WIDTH)

    def forward(self, x):
        """ Reads the relative poses out of the transformer's output tokens.
        Args:
            x: Tokens of shape (B, 1 + N * T, D), with the camera token first.
        Returns:
            The relative poses of shape (B, T - 1, 4, 4), and their log-variances of
            shape (B, T - 1, 2) ordered rotation then translation.
        """
        B, _, D = x.shape
        T = self.num_frames

        frame_descriptors = x[:, 1:].view(B, -1, T, D).mean(dim=1)
        frame_descriptors = torch.cat((x[:, :1], frame_descriptors), dim=1)

        pred = self.layers(frame_descriptors.reshape(B, -1)).view(-1, PRED_WIDTH)

        pose = torch.zeros(
            (pred.shape[0], 4, 4), device=pred.device, dtype=pred.dtype)
        pose[:, :3, :3] = roma.special_procrustes(pred[:, :ROT_DIM].view(-1, 3, 3))
        pose[:, :3, 3] = pred[:, ROT_DIM:ROT_DIM + TRANS_DIM]
        pose[:, 3, 3] = 1.

        log_variance = pred[:, ROT_DIM + TRANS_DIM:]

        return (pose.view(-1, self.prediction_dimension, 4, 4),
                log_variance.view(-1, self.prediction_dimension, CONF_DIM))

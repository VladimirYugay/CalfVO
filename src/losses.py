""" This module defines the pose loss. """
import roma
import torch
import torch.nn as nn
import torch.nn.functional as F


class PoseLoss(nn.Module):
    """ L1 translation and geodesic rotation, each under its own predicted log-variance.
        The head emits one log-variance c per edge for rotation and one for
        translation, and each term is scored as e * exp(-c) + c. Weighting the two
        terms does nothing here, since the head absorbs any weight into c.
    """

    def forward(self, model_output: dict, pose: torch.Tensor) -> torch.Tensor:
        """ Scores the predicted relative poses against the ground truth.
        Args:
            model_output: The model's relative_poses and confidences.
            pose: Ground truth relative poses, shaped like relative_poses.
        Returns:
            The mean weighted loss over every edge in the batch.
        """
        predict_pose = model_output['relative_poses']
        conf_rot = model_output['confidences'][..., 0]
        conf_trans = model_output['confidences'][..., 1]

        R, t = predict_pose[..., :3, :3], predict_pose[..., :3, 3]
        Rgt, tgt = pose[..., :3, :3], pose[..., :3, 3]

        trans_loss = F.l1_loss(t, tgt, reduction='none').mean(dim=-1)
        rot_loss = roma.rotmat_geodesic_distance(R, Rgt)

        weighted_trans_loss = (trans_loss * torch.exp(-conf_trans)) + conf_trans
        weighted_rot_loss = (rot_loss * torch.exp(-conf_rot)) + conf_rot

        return (weighted_trans_loss + weighted_rot_loss).mean()

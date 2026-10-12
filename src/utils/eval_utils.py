""" This module scores a predicted trajectory against ground truth. """
import io
from copy import deepcopy

import evo
import evo.main_ape as main_ape
import evo.main_rpe as main_rpe
import matplotlib
from evo.core.metrics import PoseRelation, Unit
from evo.core.trajectory import PoseTrajectory3D, align_trajectory
from evo.tools.plot import PlotMode
from matplotlib import pyplot as plt
from PIL import Image
from scipy.spatial.transform import Rotation

matplotlib.use('Agg')


def eval_trajectory(poses_est, poses_gt, frame_ids, align=False):
    """ Scores a predicted trajectory and draws it next to the ground truth.
        ATE and ARE are evo's absolute pose error in translation and in rotation.
        rpe_trans and rpe_rot are its relative pose error at a one frame delta over
        all pairs. With align set, every metric is computed after a Sim(3) alignment,
        which also fits the scale.
    Args:
        poses_est: Predicted camera-to-world poses of shape (N, 4, 4).
        poses_gt: Ground truth camera-to-world poses of shape (N, 4, 4).
        frame_ids: Timestamps for both trajectories.
        align: Whether to align the estimate to the reference before scoring.
    Returns:
        A dict of the four metrics, and a plot of both trajectories as a PIL image.
    """
    traj_ref = PoseTrajectory3D(
        positions_xyz=poses_gt[:, :3, 3],
        orientations_quat_wxyz=Rotation.from_matrix(
            poses_gt[:, :3, :3]).as_quat(scalar_first=True),
        timestamps=frame_ids)
    traj_est = PoseTrajectory3D(
        positions_xyz=poses_est[:, :3, 3],
        orientations_quat_wxyz=Rotation.from_matrix(
            poses_est[:, :3, :3]).as_quat(scalar_first=True),
        timestamps=frame_ids)

    ate_result = main_ape.ape(
        deepcopy(traj_ref),
        deepcopy(traj_est),
        est_name="traj",
        pose_relation=PoseRelation.translation_part,
        align=align,
        correct_scale=align)
    ate = ate_result.stats["rmse"]

    are_result = main_ape.ape(
        deepcopy(traj_ref),
        deepcopy(traj_est),
        est_name="traj",
        pose_relation=PoseRelation.rotation_angle_deg,
        align=align,
        correct_scale=align)
    are = are_result.stats["rmse"]

    rpe_rots_result = main_rpe.rpe(
        deepcopy(traj_ref),
        deepcopy(traj_est),
        est_name="traj",
        pose_relation=PoseRelation.rotation_angle_deg,
        align=align,
        correct_scale=align,
        delta=1,
        delta_unit=Unit.frames,
        rel_delta_tol=0.01,
        all_pairs=True)
    rpe_rot = rpe_rots_result.stats["mean"]

    rpe_transs_result = main_rpe.rpe(
        deepcopy(traj_ref),
        deepcopy(traj_est),
        est_name="traj",
        pose_relation=PoseRelation.translation_part,
        align=align,
        correct_scale=align,
        delta=1,
        delta_unit=Unit.frames,
        rel_delta_tol=0.01,
        all_pairs=True)
    rpe_trans = rpe_transs_result.stats["mean"]

    # Align explicitly for the plot. main_ape.ape aligns a deep copy, so traj_est is
    # still the raw estimate whatever `align` says. Same Sim(3) as the metrics above.
    traj_plot_est = align_trajectory(
        traj_est, traj_ref, correct_scale=True) if align else traj_est

    plot_mode = PlotMode.xz
    fig = plt.figure()
    ax = evo.tools.plot.prepare_axis(fig, plot_mode)
    ax.set_title(f"{'Sim(3) aligned' if align else 'unaligned'}, "
                 f"ATE {round(ate, 3)}, ARE {round(are, 3)}")
    evo.tools.plot.traj(ax, plot_mode, traj_ref, "--", "gray", "gt")
    evo.tools.plot.traj_colormap(
        ax,
        traj_plot_est,
        ate_result.np_arrays["error_array"],
        plot_mode,
        min_map=ate_result.stats["min"],
        max_map=ate_result.stats["max"],
    )
    ax.legend()

    buffer = io.BytesIO()
    plt.savefig(buffer, format='png', dpi=90)
    buffer.seek(0)

    pillow_image = Image.open(buffer)
    pillow_image.load()
    buffer.close()
    plt.close(fig)

    return {
        "ate": ate,
        "are": are,
        "rpe_rot": rpe_rot,
        "rpe_trans": rpe_trans
    }, pillow_image

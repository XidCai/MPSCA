import argparse


def str2bool(v):
    if isinstance(v, bool):
        return v
    value = v.lower()
    if value in ("true", "1"):
        return True
    if value in ("false", "0"):
        return False
    raise argparse.ArgumentTypeError("expected one of: true, false, 1, 0")

parser = argparse.ArgumentParser()


def add_argument_group(name):
    return parser.add_argument_group(name)

# -----------------------------------------------------------------------------
# Network
net_arg = add_argument_group("Network")
net_arg.add_argument(
    "--layer_num", type=int, default=5, help=""
    "number of layers. Default: 5")
net_arg.add_argument(
    "--piece_num", type=int, default=48, help=""
    "number of pieces. Default: 48")
net_arg.add_argument(
    "--decrease_layer", type=list, default=[3, 4], help=""
    "list of cluster number.")
net_arg.add_argument(
    "--cluster_top_num", type=int, default=36, help=""
    "top number of matches in cluster. Default: 36")
net_arg.add_argument(
    "--neighbor_num", type=int, default=6, help=""
    "number of neighbors. Default: 6")
net_arg.add_argument(
    "--net_channels", type=int, default=128, help=""
    "number of channels in a layer. Default: 128")
net_arg.add_argument(
    "--head", type=int, default=4, help=""
    "number of head in attention. Default: 4")
net_arg.add_argument(
    "--layer_names", type=list, default=['cluster', 'cluster', 'context', 'context', 'context', 'context', 'decluster'], help=""
    "attention layer names in each filter layer.")
net_arg.add_argument(
    "--attention_mode", type=str, default='full', help=""
    "attention mode. Default: full")
net_arg.add_argument(
    "--use_fundamental", type=str2bool, default=False, help=""
    "train fundamental matrix estimation. Default: False")
net_arg.add_argument(
    "--use_indoor", type=str2bool, default=False, help=""
    "use the indoor MPSCA network and default SUN3D checkpoint. Default: False")
net_arg.add_argument(
    "--use_ratio", type=int, choices=(0, 1, 2), default=2, help=""
    "use ratio test. 0: not use, 1: use before network, 2: use as side information. Default: 2")
net_arg.add_argument(
    "--use_mutual", type=int, choices=(0, 1, 2), default=2, help=""
    "use mutual nearest-neighbor check. 0: not use, 1: use before network, 2: use as side information. Default: 2")
net_arg.add_argument(
    "--ratio_test_th", type=float, default=0.8, help=""
    "ratio test threshold used when use_ratio=1. Default: 0.8")
net_arg.add_argument(
    "--inlier_threshold", type=float, default=0.0, help=""
    "inlier threshold. Default: 0.0")
net_arg.add_argument(
    "--use_prior_input", type=str2bool, default=True, help=""
    "use lightweight prior features as auxiliary input. Default: True")
net_arg.add_argument(
    "--use_prior_soft_bias", type=str2bool, default=True, help=""
    "use prior as soft bias in correspondence aggregation. Default: True")
net_arg.add_argument(
    "--prior_bias_scale", type=float, default=1.0, help=""
    "scale for prior soft bias in attention. Default: 1.0")
net_arg.add_argument(
    "--use_prior_logit_bias", type=str2bool, default=True, help=""
    "use corr prior as logit bias for final inlier confidence. Default: True")
net_arg.add_argument(
    "--prior_logit_bias_last_only", type=str2bool, default=True, help=""
    "apply prior logit bias only in the final LayerBlock. Default: True")
net_arg.add_argument(
    "--prior_logit_scale", type=float, default=1.0, help=""
    "scale for prior logit bias. Default: 1.0")
net_arg.add_argument(
    "--prior_use_uniqueness", type=str2bool, default=True, help=""
    "use single-image uniqueness proxy in prior input. Default: True")
net_arg.add_argument(
    "--prior_use_side_matchability", type=str2bool, default=True, help=""
    "use ratio/mutual side information as matchability prior. Default: True")
net_arg.add_argument(
    "--use_signed_prior_mask_attention", type=str2bool, default=True, help=""
    "use signed prior mask attention. Default: True")
net_arg.add_argument(
    "--signed_prior_mask_scale", type=float, default=1.0, help=""
    "scale for signed prior mask attention. Default: 1.0")

# -----------------------------------------------------------------------------
# Demo I/O
io_arg = add_argument_group("Demo I/O")
io_arg.add_argument(
    "--model_file", type=str, default=None, help=""
    "checkpoint path; defaults to the YFCC100M or SUN3D model selected by use_indoor")
io_arg.add_argument(
    "--gpu_id", type=str, default="0", help=""
    "GPU id exposed through CUDA_VISIBLE_DEVICES. Default: 0")
io_arg.add_argument(
    "--img1_path", type=str, default="./test_img1.jpg", help=""
    "first input image")
io_arg.add_argument(
    "--img2_path", type=str, default="./test_img2.jpg", help=""
    "second input image")
io_arg.add_argument(
    "--output_file", type=str, default="./inliers.jpg", help=""
    "output matching visualization")
io_arg.add_argument(
    "--num_kp", type=int, default=2000, help=""
    "maximum number of SIFT keypoints extracted per image. Default: 2000")

def get_config():
    return parser.parse_args()

#
# config.py ends here

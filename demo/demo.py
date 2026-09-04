import os
import sys
from pathlib import Path

import cv2
import numpy as np

from config import get_config


DEMO_DIR = Path(__file__).resolve().parent
CORE_DIR = DEMO_DIR.parent / "core"
if str(CORE_DIR) not in sys.path:
    sys.path.insert(0, str(CORE_DIR))


def resolve_demo_path(path):
    path = Path(path).expanduser()
    return path if path.is_absolute() else (DEMO_DIR / path).resolve()


def default_model_path(use_indoor):
    dataset = "sun3d" if use_indoor else "yfcc100m"
    return "../model/{}/model_best.pth".format(dataset)


def normalize_keypoints(image, keypoints):
    height, width = image.shape[:2]
    center = np.array([[(width - 1.0) * 0.5, (height - 1.0) * 0.5]])
    focal = max(width - 1.0, height - 1.0)
    return (keypoints - center) / focal


def compute_nn(desc1, desc2):
    import torch

    desc1 = torch.from_numpy(desc1).float()
    desc2 = torch.from_numpy(desc2).float()
    norm1 = (desc1 ** 2).sum(dim=1)
    norm2 = (desc2 ** 2).sum(dim=1)
    squared_dist = (
        norm1.unsqueeze(1)
        + norm2.unsqueeze(0)
        - 2 * torch.matmul(desc1, desc2.transpose(0, 1))
    )
    distances = squared_dist.clamp_min(0).sqrt()

    nearest_distances, nearest_in_2 = torch.topk(
        distances, k=2, dim=1, largest=False
    )
    nearest_in_2 = nearest_in_2[:, 0]
    nearest_in_1 = torch.topk(distances, k=1, dim=0, largest=False).indices.squeeze(0)

    query_indices = torch.arange(nearest_in_2.shape[0])
    mutual = (nearest_in_1[nearest_in_2] == query_indices).numpy()
    ratio = (
        nearest_distances[:, 0] / nearest_distances[:, 1].clamp_min(1e-10)
    ).numpy()
    return nearest_in_2.numpy(), ratio, mutual


def filter_initial_matches(kpts1, kpts2, ratio, mutual, opt):
    keep = np.ones(kpts1.shape[0], dtype=bool)
    if opt.use_ratio == 1:
        keep &= ratio < opt.ratio_test_th
    if opt.use_mutual == 1:
        keep &= mutual
    return kpts1[keep], kpts2[keep], ratio[keep], mutual[keep]


def draw_matching(img1, img2, points1, points2):
    height1, width1 = img1.shape[:2]
    height2, width2 = img2.shape[:2]
    visualization = np.zeros(
        (max(height1, height2), width1 + width2, 3), dtype=np.uint8
    )
    visualization[:height1, :width1] = img1
    visualization[:height2, width1:] = img2

    for point1, point2 in zip(points1, points2):
        start = tuple(np.rint(point1).astype(int))
        end = (int(round(point2[0] + width1)), int(round(point2[1])))
        cv2.line(visualization, start, end, (0, 255, 0), 1, cv2.LINE_AA)
    return visualization


class SIFTExtractor:
    def __init__(self, num_kp, contrast_threshold=1e-5):
        if hasattr(cv2, "SIFT_create"):
            self.sift = cv2.SIFT_create(
                nfeatures=num_kp, contrastThreshold=contrast_threshold
            )
        elif hasattr(cv2, "xfeatures2d") and hasattr(
            cv2.xfeatures2d, "SIFT_create"
        ):
            self.sift = cv2.xfeatures2d.SIFT_create(
                nfeatures=num_kp, contrastThreshold=contrast_threshold
            )
        else:
            raise RuntimeError("This OpenCV build does not provide SIFT.")
        self.num_kp = num_kp

    def run(self, image, image_name):
        keypoints, descriptors = self.sift.detectAndCompute(
            image.astype(np.uint8), None
        )
        if descriptors is None or not keypoints:
            raise ValueError("SIFT found no keypoints in {}.".format(image_name))

        points = np.asarray([keypoint.pt for keypoint in keypoints], dtype=np.float32)
        return points[:self.num_kp], descriptors[:self.num_kp]


def load_model(opt, model_file, device):
    import torch

    if opt.use_indoor:
        from dematch_plus_indoor import DeMatchPlus
    else:
        from dematch_plus_outdoor import DeMatchPlus

    if not model_file.is_file():
        raise FileNotFoundError("Checkpoint not found: {}".format(model_file))

    model = DeMatchPlus(opt)
    checkpoint = torch.load(str(model_file), map_location=device, weights_only=False)
    state_dict = checkpoint.get("state_dict", checkpoint)
    if not isinstance(state_dict, dict):
        raise ValueError("Checkpoint does not contain a valid state dictionary.")
    state_dict = {key.replace("module.", ""): value for key, value in state_dict.items()}
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()
    return model


def demo(opt):
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    img1_path = resolve_demo_path(opt.img1_path)
    img2_path = resolve_demo_path(opt.img2_path)
    model_file = resolve_demo_path(
        opt.model_file or default_model_path(opt.use_indoor)
    )
    output_file = resolve_demo_path(opt.output_file)

    img1 = cv2.imread(str(img1_path))
    img2 = cv2.imread(str(img2_path))
    if img1 is None or img2 is None:
        raise FileNotFoundError(
            "Could not load input images: {} and {}".format(img1_path, img2_path)
        )

    extractor = SIFTExtractor(num_kp=opt.num_kp)
    kpts1, desc1 = extractor.run(img1, img1_path)
    kpts2_all, desc2 = extractor.run(img2, img2_path)
    if desc2.shape[0] < 2:
        raise ValueError("The second image must provide at least two SIFT descriptors.")

    nearest_in_2, ratio, mutual = compute_nn(desc1, desc2)
    kpts2 = kpts2_all[nearest_in_2]
    kpts1, kpts2, ratio, mutual = filter_initial_matches(
        kpts1, kpts2, ratio, mutual, opt
    )

    minimum_matches = max(8, opt.neighbor_num)
    if kpts1.shape[0] < minimum_matches:
        raise ValueError(
            "Only {} putative matches remain; at least {} are required.".format(
                kpts1.shape[0], minimum_matches
            )
        )

    xs = np.concatenate(
        [normalize_keypoints(img1, kpts1), normalize_keypoints(img2, kpts2)],
        axis=-1,
    )
    data = {
        "xs": torch.from_numpy(xs).float().unsqueeze(0).unsqueeze(1).to(device)
    }

    side_features = []
    if opt.use_ratio == 2:
        side_features.append(ratio[:, None])
    if opt.use_mutual == 2:
        side_features.append(mutual.astype(np.float32)[:, None])
    if side_features:
        sides = np.concatenate(side_features, axis=-1)
        data["sides"] = torch.from_numpy(sides).float().unsqueeze(0).to(device)

    model = load_model(opt, model_file, device)
    with torch.inference_mode():
        logits_list, _ = model(data, training=False)
    scores = logits_list[-1][0].cpu().numpy()
    retained = scores > opt.inlier_threshold

    matching = draw_matching(img1, img2, kpts1[retained], kpts2[retained])
    output_file.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_file), matching):
        raise IOError("Failed to write output image: {}".format(output_file))

    variant = "indoor" if opt.use_indoor else "outdoor"
    print("Device: {} | model: {}".format(device, variant))
    print("Putative matches: {} | retained: {}".format(len(scores), retained.sum()))
    print("Output: {}".format(output_file))


if __name__ == "__main__":
    demo(get_config())

import numpy as np
import torch
import os
import sys
from tqdm import tqdm
from dataset import collate_fn, CorrespondencesDataset
from utils import compute_pose_error, pose_auc, estimate_pose_norm_kpts, estimate_pose_from_E
from config import get_config

sys.path.append('../core')

torch.set_grad_enabled(False)
torch.manual_seed(0)


def get_prior_stats(model):
    model_ref = model.module if hasattr(model, 'module') else model
    return getattr(model_ref, 'latest_prior_stats', {})


def use_non_strict_load(config):
    return config.use_prior_input or config.use_prior_soft_bias or config.use_prior_logit_bias


def inlier_test(config, polar_dis, inlier_mask):
    polar_dis = polar_dis.reshape(inlier_mask.shape).unsqueeze(0)
    inlier_mask = torch.from_numpy(inlier_mask).type(torch.float32)
    is_pos = (polar_dis < config.obj_geod_th).type(inlier_mask.type())
    is_neg = (polar_dis >= config.obj_geod_th).type(inlier_mask.type())
    precision = torch.mean(
        torch.sum(inlier_mask * is_pos, dim=1) /
        (torch.sum(inlier_mask * (is_pos + is_neg), dim=1)+1e-15)
    )
    recall = torch.mean(
        torch.sum(inlier_mask * is_pos, dim=1) /
        torch.sum(is_pos, dim=1)
    )
    f_scores = 2*precision*recall/(precision+recall+1e-15)

    return precision, recall, f_scores

def test(config):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Running inference on device \"{}\"'.format(device))
    if config.use_indoor:
        from dematch_plus_indoor import DeMatchPlus
    else:
        from dematch_plus_outdoor import DeMatchPlus
    print('Model variant: {}'.format('indoor' if config.use_indoor else 'outdoor'))

    test_dataset = CorrespondencesDataset(config.data_te, config)
    test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=1, shuffle=False,
            num_workers=8, pin_memory=True, collate_fn=collate_fn)

    model = DeMatchPlus(config)

    save_file_best = os.path.join(config.model_file, "model_best.pth")
    if not os.path.exists(save_file_best):
        print("Model File {} does not exist! Quiting".format(save_file_best))
        exit(1)
    # Restore model
    checkpoint = torch.load(save_file_best, map_location=device, weights_only=False)
    state_dict = {k.replace('module.', ''): v for k, v in checkpoint['state_dict'].items()}
    model.load_state_dict(state_dict, strict=not use_non_strict_load(config))
    model.to(device)
    model.eval()

    err_ts, err_Rs = [], []
    precision_all, recall_all, f_scores_all = [], [], []
    prior_info = {}
    for index, test_data in enumerate(tqdm(test_loader)):
        x = test_data['xs'].to(device)
        y = test_data['ys'].to(device)
        R_gt, t_gt = test_data['Rs'], test_data['ts']

        data = {}
        data['xs'] = x
        if 'sides' in test_data and torch.is_tensor(test_data['sides']):
            data['sides'] = test_data['sides'].to(device)
        logits_list, e_hat_list = model(data, training=False)
        for key, value in get_prior_stats(model).items():
            prior_info.setdefault(key, []).append(value)
        logits = logits_list[-1]
        e_hat = e_hat_list[-1].cpu().detach().numpy().reshape(3,3)

        mkpts0 = x.squeeze()[:, :2].cpu().detach().numpy()
        mkpts1 = x.squeeze()[:, 2:].cpu().detach().numpy()
        inlier_mask = logits.squeeze().cpu().detach().numpy() > config.inlier_threshold
        mask_kp0 = mkpts0[inlier_mask]
        mask_kp1 = mkpts1[inlier_mask]

        if config.use_ransac == True:
            ret = estimate_pose_norm_kpts(mask_kp0, mask_kp1, conf=config.ransac_prob)
        else:
            if e_hat.shape[0] == 0:
                print("Algorithm has no essential matrix output, can not eval without robust estimator such as RANSAC.")
                print("Try to set use_ransac=True in config file.")
                exit(1)
            ret = estimate_pose_from_E(mkpts0, mkpts1, inlier_mask, e_hat)
        if ret is None:
            err_t, err_R = np.inf, np.inf
            precision_all.append(0.0)
            recall_all.append(0.0)
            f_scores_all.append(0.0)
        else:
            R, t, inlier_mask_new = ret
            T_0to1 = torch.cat([R_gt.squeeze(), t_gt.squeeze().unsqueeze(-1)], dim=-1).numpy()
            err_t, err_R = compute_pose_error(T_0to1, R, t)

        err_ts.append(err_t)
        err_Rs.append(err_R)

        precision, recall, f_scores = inlier_test(config, y, inlier_mask)
        precision_all.append(float(precision))
        recall_all.append(float(recall))
        f_scores_all.append(float(f_scores))

    out_eval = {'error_t': err_ts,
                'error_R': err_Rs}

    pose_errors = []
    for idx in range(len(out_eval['error_t'])):
        pose_error = np.maximum(out_eval['error_t'][idx], out_eval['error_R'][idx])
        pose_errors.append(pose_error)

    thresholds = [5, 10, 20]
    aucs = pose_auc(pose_errors, thresholds)
    aucs = [100.*yy for yy in aucs]
    precision, recall, f_scores = np.mean(np.asarray(precision_all))*100, np.mean(np.asarray(recall_all))*100, np.mean(np.asarray(f_scores_all))*100

    print('Evaluation Results (mean over {} pairs):'.format(len(test_loader)))
    print('AUC@5\t AUC@10\t AUC@20\t')
    print('{:.2f}\t {:.2f}\t {:.2f}\t'.format(aucs[0], aucs[1], aucs[2]))
    print('Prec\t Rec\t F1\t')
    print('{:.2f}\t {:.2f}\t {:.2f}\t'.format(precision, recall, f_scores))

    return

if __name__ == '__main__':
    config, unparsed = get_config()
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = config.gpu_id
    test(config)

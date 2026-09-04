import torch
import torch.optim as optim
from tqdm import trange
import os
from valid import valid
from loss import MatchLoss
from utils import tocuda
try:
    from tensorboardX import SummaryWriter
except ImportError:
    try:
        from torch.utils.tensorboard import SummaryWriter
    except ImportError:
        class SummaryWriter:
            def __init__(self, *args, **kwargs):
                print('TensorBoard logging is disabled: no tensorboardX/tensorboard found.')

            def add_scalar(self, *args, **kwargs):
                pass

            def add_text(self, *args, **kwargs):
                pass
# tensorboard --logdir=../model --host=127.0.0.1 --port=6006


def get_prior_stats(model):
    model_ref = model.module if hasattr(model, 'module') else model
    return getattr(model_ref, 'latest_prior_stats', {})


def use_non_strict_load(config):
    return config.use_prior_input or config.use_prior_soft_bias or config.use_prior_logit_bias


def log_prior_config(config, writer):
    writer.add_text('prior/config', (
        'use_prior_input={}\n'
        'use_prior_soft_bias={}\n'
        'prior_use_uniqueness={}\n'
        'use_prior_logit_bias={}\n'
        'prior_logit_bias_last_only={}\n'
        'use_signed_prior_mask_attention={}\n'
        'signed_prior_mask_scale={}\n'
        'prior_use_side_matchability={}\n'
        'prior_bias_scale={}\n'
        'prior_logit_scale={}\n'
        'use_ratio={}\n'
        'use_mutual={}\n'
        'loss_essential={}\n'
        'loss_essential_init_iter={}'
    ).format(
        config.use_prior_input,
        config.use_prior_soft_bias,
        config.prior_use_uniqueness,
        config.use_prior_logit_bias,
        config.prior_logit_bias_last_only,
        config.use_signed_prior_mask_attention,
        config.signed_prior_mask_scale,
        config.prior_use_side_matchability,
        config.prior_bias_scale,
        config.prior_logit_scale,
        config.use_ratio,
        config.use_mutual,
        config.loss_essential,
        config.loss_essential_init_iter,
    ), 0)


def train_step(step, optimizer, model, match_loss, data):
    model.train()
    if step >= 80000:
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr']*0.999996

    res_logits, res_e_hat = model(data, training=True)
    loss = 0
    loss_val = []
    for i in range(len(res_logits)):
        loss_i, geo_loss, cla_loss, l2_loss, _, _ = match_loss.run(step, data, res_logits[i], res_e_hat[i])
        loss += loss_i
        loss_val += [geo_loss, cla_loss, l2_loss]
    optimizer.zero_grad()
    loss.backward()
    for name, param in model.named_parameters():
        if param.grad is not None and torch.any(torch.isnan(param.grad)):
            print('skip because nan')
            return loss_val, get_prior_stats(model)

    optimizer.step()
    return loss_val, get_prior_stats(model)


def train(model, train_loader, valid_loader, config):
    model.cuda()
    optimizer = optim.Adam(model.parameters(), lr=config.train_lr, weight_decay = config.weight_decay)
    match_loss = MatchLoss(config)

    checkpoint_path = os.path.join(config.log_path, 'checkpoint.pth')
    config.resume = os.path.isfile(checkpoint_path)
    writer=SummaryWriter(os.path.join(config.log_path, 'log_file'))
    print('Prior config: use_prior_input={}, use_prior_soft_bias={}, prior_use_uniqueness={}, '
          'use_prior_logit_bias={}, prior_logit_bias_last_only={}, prior_use_side_matchability={}, '
          'use_signed_prior_mask_attention={}, signed_prior_mask_scale={}, '
          'prior_bias_scale={}, prior_logit_scale={}, use_ratio={}, use_mutual={}, '
          'loss_essential={}, loss_essential_init_iter={}'.format(
              config.use_prior_input,
              config.use_prior_soft_bias,
              config.prior_use_uniqueness,
              config.use_prior_logit_bias,
              config.prior_logit_bias_last_only,
              config.prior_use_side_matchability,
              config.use_signed_prior_mask_attention,
              config.signed_prior_mask_scale,
              config.prior_bias_scale,
              config.prior_logit_scale,
              config.use_ratio,
              config.use_mutual,
              config.loss_essential,
              config.loss_essential_init_iter,
          ))
    print('Late snapshots: enabled={}, start={}, interval={}'.format(
        config.save_late_snapshots,
        config.late_snapshot_start,
        config.val_intv,
    ))
    log_prior_config(config, writer)
    if config.resume:
        print('==> Resuming from checkpoint..')
        checkpoint = torch.load(checkpoint_path, weights_only=False)
        best_acc = checkpoint['best_acc']
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'], strict=not use_non_strict_load(config))
        optimizer.load_state_dict(checkpoint['optimizer'])
    else:
        best_acc = -1
        start_epoch = 0
    train_loader_iter = iter(train_loader)
    for step in trange(start_epoch, config.train_iter, ncols=config.tqdm_width):
        try:
            train_data = next(train_loader_iter)
        except StopIteration:
            train_loader_iter = iter(train_loader)
            train_data = next(train_loader_iter)
        train_data = tocuda(train_data)

        # run training
        cur_lr = optimizer.param_groups[0]['lr']
        loss_vals, prior_stats = train_step(step, optimizer, model, match_loss, train_data)
        if step%config.log_intv==0:
            writer.add_scalar('lr', cur_lr, step)
            writer.add_scalar('EssentionLoss', loss_vals[0], step)
            writer.add_scalar('ClassifyLoss', loss_vals[1], step)
            writer.add_scalar('RegressionLoss', loss_vals[2], step)
            for key, value in prior_stats.items():
                writer.add_scalar(key, value, step)

        # Check if we want to write validation
        b_save = ((step + 1) % config.save_intv) == 0
        b_validate = ((step + 1) % config.val_intv) == 0
        if b_validate:
            va_res, geo_loss, cla_loss, l2_loss,  _, _, _, valid_prior_stats = valid(valid_loader, model, step, config)
            writer.add_scalar('val_EssentionLoss', geo_loss, step)
            writer.add_scalar('val_ClassifyLoss', cla_loss, step)
            writer.add_scalar('val_RegressionLoss', l2_loss, step)
            writer.add_scalar('val_acc', va_res, step)
            for key, value in valid_prior_stats.items():
                writer.add_scalar('val_' + key, value, step)
            if va_res > best_acc:
                print("Saving best model with va_res = {}".format(va_res))
                best_acc = va_res
                torch.save({
                'epoch': step + 1,
                'state_dict': model.state_dict(),
                'best_acc': best_acc,
                'optimizer' : optimizer.state_dict(),
                }, os.path.join(config.log_path, 'model_best.pth'))

            if config.save_late_snapshots and (step + 1) >= config.late_snapshot_start:
                snapshot_dir = os.path.join(
                    config.log_path,
                    'late_snapshots',
                    'step_{:06d}'.format(step + 1),
                )
                os.makedirs(snapshot_dir, exist_ok=True)
                snapshot_path = os.path.join(snapshot_dir, 'model_best.pth')
                torch.save({
                    'epoch': step + 1,
                    'state_dict': model.state_dict(),
                    'best_acc': best_acc,
                    'va_res': va_res,
                }, snapshot_path)
                print("Saving late snapshot to {} with va_res = {}".format(
                    snapshot_path,
                    va_res,
                ))

        if b_save:
            torch.save({
            'epoch': step + 1,
            'state_dict': model.state_dict(),
            'best_acc': best_acc,
            'optimizer' : optimizer.state_dict(),
            }, checkpoint_path)

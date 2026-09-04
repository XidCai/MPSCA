# MPSCA is derived from DeMatch++ and includes code borrowed from OANet:
# https://github.com/zjhthu/OANet
# Original DeMatch++ implementation: Shihua Zhang, 2024
# MPSCA modifications: XidCai, 2026


from config import get_config, print_usage
config, unparsed = get_config()
import os
os.environ['CUDA_VISIBLE_DEVICES'] = config.gpu_id
import torch.utils.data
import sys
from data import collate_fn, CorrespondencesDataset
if config.use_indoor:
    from dematch_plus_indoor import DeMatchPlus as Model
else:
    from dematch_plus_outdoor import DeMatchPlus as Model
from train import train


print("---------------------------MPSCA---------------------------")
print("Note: To combine datasets, use .")
print("Model variant: {}".format("indoor" if config.use_indoor else "outdoor"))

def create_log_dir(config):
    os.makedirs(config.log_base, exist_ok=True)
    if config.log_suffix == "":
        suffix = "-".join(sys.argv)
    else:
        suffix = config.log_suffix
    result_path = os.path.join(config.log_base, suffix)
    os.makedirs(os.path.join(result_path, "train"), exist_ok=True)
    os.makedirs(os.path.join(result_path, "valid"), exist_ok=True)
    config_path = os.path.join(result_path, "config.th")
    if os.path.exists(config_path):
        print('warning: will overwrite config file')
    torch.save(config, config_path)
    config.log_path = result_path

def main(config):
    """The main function."""

    # Initialize network
    model = Model(config)

    # Run propper mode
    create_log_dir(config)

    train_dataset = CorrespondencesDataset(config.data_tr, config)

    train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=config.train_batch_size, shuffle=True,
            num_workers=16, pin_memory=False, collate_fn=collate_fn)

    valid_dataset = CorrespondencesDataset(config.data_va, config)
    valid_loader = torch.utils.data.DataLoader(
            valid_dataset, batch_size=config.train_batch_size, shuffle=False,
            num_workers=8, pin_memory=False, collate_fn=collate_fn)
    #valid_loader = None
    print('start training .....')
    train(model, train_loader, valid_loader, config)


if __name__ == "__main__":
    # Parse configuration
    config, unparsed = get_config()
    # If we have unparsed arguments, print usage and exit
    if len(unparsed) > 0:
        print_usage()
        exit(1)

    main(config)

#
# main.py ends here

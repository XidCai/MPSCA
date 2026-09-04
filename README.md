# MPSCA

MPSCA is a prior-guided method for two-view correspondence learning and outlier rejection. It extends a lightweight correspondence prior and signed clustering attention while retaining the original motion-field decomposition and recovery framework. 

## Environment

The code needs to test with Python 3.10, PyTorch 2.7.1, and CUDA 11.8. Create an environment and install the dependencies from the repository root:

```bash
conda create -n mpsca python=3.10 -y
conda activate mpsca
pip install -r requirements.txt
```

If the PyTorch build in `requirements.txt` does not match your CUDA installation, install a compatible PyTorch build first and then install the remaining dependencies.

## Architecture and weights

Arrange the precomputed correspondence files and trained checkpoints as follows:

```text
MPSCA/
|-- core/
|-- demo/
|-- test/
|-- data_dump/
|   |-- yfcc-sift-2000-train.hdf5
|   |-- yfcc-sift-2000-test.hdf5
|   |-- sun3d-sift-2000-train.hdf5
|   |-- sun3d-sift-2000-test.hdf5
|-- model/
    |-- yfcc100m/
    |   |-- model_best.pth
    |-- sun3d/
        |-- model_best.pth
```

Then generate matches for YFCC100M and SUN3D  with SIFT.

```bash
cd ./dump_match
python extract_feature.py
python yfcc.py

python extract_feature.py --input_path=../raw_data/sun3d_test
python sun3d.py
```

Download the pretrained MPSCA checkpoints separately:

- [YFCC100M model](https://github.com/XidCai/MPSCA/releases/download/v1.0.0/yfcc100m.tar.gz)
- [SUN3D model](https://github.com/XidCai/MPSCA/releases/download/v1.0.0/sun3d.tar.gz)

Extract the downloaded archives from the repository root:

```bash
mkdir -p model
tar -xzf yfcc100m.tar.gz -C model
tar -xzf sun3d.tar.gz -C model
```


## Demo


```bash
cd demo
python demo.py
```

The visualization is written to `demo/inliers.jpg`.

## Train

After preparing the YFCC100M training and validation data, run the training script:

```bash
cd ./core
python main.py
```


## Evaluation

Run evaluation commands from `test/`. 

```bash
cd test
python test.py
```


## Acknowledgement

This code is built upon [DeMatch++](https://github.com/SuhZhang/DeMatchPlus) and [OANet](https://github.com/zjhthu/OANet). If you use this repository, please cite the following papers:

```bibtex
@article{zhang2025dematchplusplus,
  title={{DeMatch++}: Two-View Correspondence Learning via Deep Motion Field Decomposition and Respective Local-Context Aggregation},
  author={Zhang, Shihua and Li, Zizhuo and Ma, Jiayi},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
  volume={47},
  number={12},
  pages={11234--11251},
  year={2025},
  doi={10.1109/TPAMI.2025.3596598}
}

@inproceedings{zhang2019oanet,
  title={Learning Two-View Correspondences and Geometry Using Order-Aware Network},
  author={Zhang, Jiahui and Sun, Dawei and Luo, Zixin and Yao, Anbang and Zhou, Lei and Shen, Tianwei and Chen, Yurong and Quan, Long and Liao, Hongen},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={5845--5854},
  year={2019}
}
```

## License

This project is released under the MIT License. See [LICENSE](LICENSE) for details.

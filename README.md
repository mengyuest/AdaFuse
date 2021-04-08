# AdaFuse

## Dependencies
Red Hat 7.6, Tesla-V100x4

(Here shows how to install the environment using `conda`)
```bash
conda create --name newenv python=3.7.6
conda activate newenv
conda install pytorch=1.2.0 torchvision
pip install thop==0.0.31-2001170342
```

## Preparations
Here taking SomethingV1 dataset for an example:
1. Download all the tar-filed video frames from [https://20bn.com/datasets/something-something/v1](https://20bn.com/datasets/something-something/v1) and  untar them by `cat 20bn-something-something-v1-?? | tar zx`
2. In `common.py`,  modify `ROOT_DIR`, `DATA_PATH` and `EXPS_PATH` to setup the dataset path and logs path (where to save checkpoints and logs)

## Evaluation on SomethingV1
Here we are trying to reproduce the results for `AdaFuse-TSN50`, `AdaFuse-TSM50` and `AdaFuse-TSM50Last` as shown in Table 3 in the main paper for SomethingV1 dataset.
1. Download the pretrained models from [https://drive.google.com/drive/folders/1riSu-E0EzvpLHha5tcbC9OoAAX5i3cBL?usp=sharing](https://drive.google.com/drive/folders/1riSu-E0EzvpLHha5tcbC9OoAAX5i3cBL?usp=sharing) and put the folder under your experiment directory `EXPS_PATH`
2. Run `sh zz_eval.sh` and you will see the expected/real results (4 GPUS are needed to guarantee the exact results)

## Training on SomethingV1
To train `AdaFuse-TSM50Last` on SomethingV1 dataset, simply run: `sh zz_train.sh`. It takes less than 14 hours on 4 TeslaV100 GPUs and should get similar accuracy (~ 46.8%) as shown in Table 3 in the main paper.


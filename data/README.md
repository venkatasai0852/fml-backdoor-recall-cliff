# Data

No dataset or model checkpoint is stored in this repository.

## CIFAR-10

`core_kaggle.py` loads CIFAR-10 in this order:

1. `/kaggle/input/**/cifar-10-python.tar.gz` or the extracted `data_batch_1`
   files when running on Kaggle;
2. `torchvision.datasets.CIFAR10` with `download=True` otherwise.

The frozen split is the standard 50,000 training and 10,000 test images. Target
class 0, poison rate 5% (10% in the amplification study).

## RadioML 2016.10a

The radio study needs `RML2016.10a_dict.pkl`. It is not redistributed because of
its size and license. On Kaggle, add the public dataset:

```text
Add Input -> Datasets -> radioml2016-deepsigcom -> Add
```

The notebook searches, in order:

```text
/kaggle/input/**/RML2016.10a_dict.pkl
/kaggle/input/**/*.pkl
./RML2016.10a_dict.pkl
./data/RML2016.10a_dict.pkl
```

Download the dataset from DeepSig and observe its license. The study uses
11 modulation classes, SNR from -20 to 18 dB in 2 dB steps, and 2x128 I/Q
windows with unit-energy normalisation.

## Checkpoints

Model checkpoints (`*.pt`, `*.pth`, `*.pkl`) are ignored by git. They are large
and reproducible from the drivers; the result CSVs needed to rebuild every
table and figure are committed.

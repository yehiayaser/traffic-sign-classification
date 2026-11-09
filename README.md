# Traffic Sign Classification

Compare a small CNN trained from scratch with a ResNet-18 initialized from ImageNet weights on GTSRB. The input is a cropped RGB image containing one sign. The output is one of 43 classes.

## Experiment

Both models use the same image size (96 × 96), image normalization, train/validation split, held-out test set, and number of epochs. Each model uses its own learning rate. The comparison measures the two training recipes. It does not isolate the effect of pretraining.

GTSRB contains groups of photos of the same physical sign. We keep each group entirely in training or validation and reserve the official test split for final evaluation. The checkpoint with the highest validation macro-F1 is evaluated on test once per model.

## Run

Use Python 3.10 or newer and a matching pair of PyTorch and torchvision builds. On Kaggle or Colab, select a GPU runtime, enable dataset and weight downloads, and upload the notebook and train.py. The notebook explains what to inspect in each step.

```shell
python -m pip install -r requirements.txt
python train.py --epochs 5 --batch-size 64 --output runs/baseline
```

Use a fresh output directory for each run. If GPU memory is tight, try `--batch-size 32`. The script saves each model's best checkpoint, a confusion matrix CSV, and a JSON report with macro-F1, accuracy, epoch timings, and the five most frequent class confusions. Save the run folder outside the cloud session if you want to keep the artifacts.

## Questions for the first run

- Follow the baseline shapes from RGB input through both pooling layers to the 43 logits.
- Compare macro-F1 with accuracy. Which result reveals more about the classes with fewer examples?
- Inspect the five largest off-diagonal confusion counts. What sign features might explain those errors?

Dataset source: [German Traffic Sign Recognition Benchmark](https://benchmark.ini.rub.de/gtsrb_dataset.html).

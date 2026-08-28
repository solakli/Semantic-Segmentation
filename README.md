# Ship Detection and Segmentation

Pixel-level segmentation of ships in satellite imagery with a U-Net, trained on the
[Airbus Ship Detection](https://www.kaggle.com/c/airbus-ship-detection/data) dataset
(~109,000 images) on an AWS `p2.8xlarge` instance.

USC EE599, Deep Learning, May 2019. Team project with Darun Jayavel and Arushi Kapurwan.

## Result

| Model | Input size | Channels | Batch norm | Time / epoch | IoU |
|---|---|---|---|---|---|
| Baseline U-Net | 768x768 | 8, 16, 32, 64 | no | 4 h | 0.05 |
| Revised U-Net | 384x384 | 12, 24, 48, 96 | yes | 1.5 h | 0.20 |

The interesting part of this project was not the final score, which is low. It was
figuring out why the baseline failed and what to change.

## What went wrong first, and why

The baseline segmented cleanly when the water was clear, and fell apart on images with
clouds, land, or coastline in the frame. Small ships against a busy background were
missed almost entirely. Shrinking the batch size and adjusting the learning rate did not
help, which pointed at capacity rather than optimization: with only 8 channels in the
first block, the filters could not separate a small bright object from the noise around it.

Three changes followed from that:

1. **More channels per block** (8/16/32/64 to 12/24/48/96) so the early layers have enough
   filters to represent ships and clutter separately.
2. **Batch normalization** after every down and up block. The original U-Net paper predates
   common use of it, so the reference architecture has none.
3. **Downsizing inputs to 384x384.** A wider model at full resolution did not fit the
   memory or time budget. Halving the input paid for the extra channels and cut training
   from 4 hours to 1.5 hours per epoch.

Together those took IoU from 0.05 to 0.20 in 2 epochs. Both numbers are low in absolute
terms, and the honest reason is budget: this ran on paid EC2 time for a course, so the
final model only got 2 epochs. The trend across the change, not the endpoint, is the result.

## Data

The raw set is about 4:1 empty water to images containing ships. Training on that ratio
rewards a model for predicting "no ship" everywhere. I rebalanced to 3:2, giving 109,000
training images at 60% empty and 40% containing one or more ships.

Augmentation used the Keras image generator plus a custom function: pixel normalization
by 255, horizontal and vertical flips, brightness jitter, and resizing.

## Architecture

Standard U-Net, 3 down blocks and 3 up blocks. Each down block is 2 3x3 convolutions,
ReLU, then 2x2 max pooling, with channels doubling per block. A bottleneck of 2
convolutions with no pooling sits in the middle. The up path uses transposed convolutions
to double the spatial size, concatenated with the matching down-block output as a skip
connection. All convolutions use full padding so ships touching the image edge are not cut off.

Loss functions tried: Dice, IoU, and cross-entropy. Optimizers: Adam and SGD with momentum.
Batch sizes swept over [4, 8, 16, 32], learning rates over [0.0001, 0.0005, 0.001, 0.004].

## Files

| File | What it does |
|---|---|
| `Ship_Detection/dl_384.py` | Model definition and training loop for the 384x384 model |
| `Ship_Detection/Data_Augmentation.py` | Custom augmentation pipeline |
| `Ship_Detection/Data_exploration.py` | Dataset stats and class-balance analysis |
| `Ship_Detection/Test.py` | Inference and IoU scoring |
| `Ship_Detection/Report.pdf` | Full write-up with predicted masks and per-layer activation maps |

## If I did this again

Longer training on the final architecture, since 2 epochs is nowhere near convergence.
A pretrained encoder (ResNet or EfficientNet) instead of training the down path from
scratch. And a two-stage setup, classify ship / no ship first and only segment the
positives, which is what the top Kaggle solutions did and which suits a set that is 60%
empty water.

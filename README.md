# intermediate-projection-generation-on-CAT
intermediate porjection generation on for CT-imaging using deep learning


## To run training

1. Install environment
2. Create a exp_config.yml file to define your experiment
3. run `python -m src.train`
4. _Optionally_, run `tensorboard --logdir 'outputs'` to see logs for all experiments

## Batch sizes
These are the batch sizes that best utilize a 16GB GPU
|             | train size  | validation size  |
|:-----------:|:-----------:|:----------------:|
| autoencoder |      14     |        53        |
|     unet    |      12     |        24        |
|    cunet    |      12     |        24        |
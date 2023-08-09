

# Response Forecasting for New Media

The repository for [Measuring the Effect of Influential Messages on Varying Personas](https://arxiv.org/pdf/2305.16470.pdf) (ACL 2023).


## Environments
- Ubuntu-18.0.4
- Python (3.7)
- Cuda (11.1)

## Installation
Install [Pytorch](https://pytorch.org/) 1.9.0, then run the following in the terminal:
```shell
cd MRFP # get into MRFP folder
conda create -n respred python=3.7 -y  # create a new conda environment
conda activate respred

chmod +x scripts/setup.sh
./scripts/setup.sh
```

Due to privacy reason, we release data pointers for the data used in the paper. Please download data from [here](https://drive.google.com/drive/folders/1rL8DRzre-wkCc8Pa7xZfhwSpgmVfbzRE?usp=sharing) and use the up-to-date Twitter API to fetch data and populate the dictionaries in each file. After populating the data files, then move the files into `twitter_crawl/data_new2/CNN/`.

## Note
The running of the system might require [wandb](wandb.ai) account login

## Train Models
To retrain the models, edit the parameters in the following and run in the terminal.

Arguments: --plm: choose from `t5-base`, `bart-base`, `gpt2`, --load_model_path: either the wandb run id like `abcd1234` or the model file path

```shell
#eval 
python main.py \
    --use_cache 0 \
    --batch_size 16 \
    --eval_batch_size 16 \
    --plm $plm \
    --plm_lr 5.e-5 \
    --num_beams 1 \
    --num_epochs 20 \
    --config response_pred_by_gen

#eval 
python main.py \
    --use_cache 0 \
    --eval_batch_size 6 \
    --no_dl_score 0 \
    --plm $plm \
    --load_model_path $model_path  \
    --config response_pred_by_gen_eval
```


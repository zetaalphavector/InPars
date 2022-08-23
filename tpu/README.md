# Automated download, train and evaluation on a GCP TPUv3

## Download

Let's first download all the data which are:

* Synthetic queries
* Pyserini BM25 runs
* Qrels
* Queries
* Corpus
  
Be sure to have at least 350GB available on disk and follow this [guide](https://github.com/castorini/pygaggle/blob/master/docs/experiments-monot5-tpu.md#setup-environment-on-vm) to install T5 dependencies.

```
pip install -U pyserini
pip install wget ir-measures
cd tpu/
nohup python -u download_data.py &
```

# Training on synthetic data
Use TPUv3 to train on each dataset
```
nohup python -u train_inpars.py \
    --gcp_path path_to_gcp_bucket \
    --tpu_proj project_name \
    --tpu_name tpu_name &
```

# Retrieval
Use TPUv3 to retrieve on each dataset
```
nohup python run_t5_3B_inpars.py \
    --gcp_path path_to_gcp_bucket \
    --tpu_proj project_name \
    --tpu_name tpu_name &
```

# Download and evaluate

Download retrieval's scores for each dataset and evaluate:

```
nohup python get_t5_3B_inpars.py \
    --gcp_path path_to_gcp_bucket \
    --tpu_proj project_name \
    --tpu_name tpu_name &
```

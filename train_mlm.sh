#!/bin/bash

python run_mlm.py  --model_name_or_path bert-base-uncased\
  --dataset_name wikitext\
  --dataset_config_name wikitext-2-raw-v1\
  --do_train\
  --do_eval\
  --output_dir output

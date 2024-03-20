## Run Medcontext on BTCV

```bash
python main.py --json_list dataset_synapse_split.json --val_every 100 --batch_size=1 --feature_size=32 --rank 0 --logdir=PATH/TO/OUTPUT/FOLDER --optim_lr=1e-4 --lrschedule=warmup_cosine --infer_overlap=0.5 --save_checkpoint --data_dir=YOUR/DATA/DIRECTORY/CONTAINING/JSON/FILE
```

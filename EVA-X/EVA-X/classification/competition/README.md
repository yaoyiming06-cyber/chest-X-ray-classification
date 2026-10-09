# EVA-X competition adapter

This adapter keeps competition data handling separate from the CheXpert loader.
Its normalized manifest is an internal format; convert official files into it
only after checking the real image layout, label semantics, and sample template.

## Manifest

Training and validation CSVs contain one row per image and these columns:

```text
image_path,Subject_id,Study_id,label_0,label_1,label_2,label_3,label_4,label_5,label_6,label_7,label_8,label_9
```

`image_path` is relative to `--data_path` or absolute. Label values `1` and `0`
are preserved; empty, `NaN`, `None`, and `null` values map to `0`; `-1` is
preserved for strategy-specific handling. Per-class
uncertainty strategies follow `labels.py`: U-Ones for Edema and Atelectasis;
U-MultiClass for Enlarged Cardiomediastinum, Cardiomegaly, and Pleural
Effusion; U-SelfTrained for Pneumothorax, Consolidation, and Pneumonia;
U-Ignore for Lung Opacity; and ordinary binary classification for No Finding.
U-MultiClass gets a three-way `{negative, positive, uncertain}` head and
cross-entropy loss; reported positive probability is `p(positive) / (p(negative)
+ p(positive))`, as in the CheXpert paper. U-Ignore masks uncertain entries from
BCE. U-SelfTrained uses the same first-stage masked BCE, then replaces only its
uncertain entries with soft Study-level probabilities from the first-stage
checkpoint. U-Ones maps `-1` to `1`; optional `--competition_ones_lsr 0.1`
smooths those binary targets. Every image in one Study must have identical
labels. Training samples one image per Study per epoch; evaluation visits every
image and averages logits within that Study.

## Inspect downloaded data

From `classification/`:

```powershell
python competition/inspect_data.py D:\path\to\competition-data
python competition/inspect_data.py D:\path\to\competition-data --train-csv train.csv --valid-csv val.csv
```

The inspector prints file types, table headers and examples, row counts,
Study multiplicity, class counts, and optional Subject overlap.

## Train

After preparing the normalized manifests, start with a single GPU, 224-pixel
inputs, a small batch, AMP from the existing EVA-X scaler, and gradient
accumulation. The organizer has confirmed EVA-X generic self-supervised
pretraining is allowed.

```powershell
python train.py --dataset competition --nb_classes 10 --model eva02_small_patch16_xattn_fusedLN_SwiGLU_preln_RoPE --input_size 224 --batch_size 2 --accum_iter 8 --num_workers 2 --epochs 10 --eval_interval 1 --data_path D:\path\to\competition-data --train_list D:\path\to\train_manifest.csv --test_list D:\path\to\val_manifest.csv --finetune D:\path\to\eva_x_small_patch16_merged520k_mim.pt --output_dir .\output\competition\seed42 --log_dir .\output\competition\seed42 --seed 42 --use_mean_pooling
```

The initial training transform preserves the full image by padding to square,
then applies only a small affine perturbation and mild brightness/contrast
changes on training images. Validation uses deterministic preprocessing.

For U-SelfTrained, first train the model with no pseudo-label arguments. Run it
on the training manifest to produce Study-level probabilities, then initialize
a fresh second-stage optimizer from that checkpoint and supply both files:

```powershell
python -m competition.predict --manifest D:\path\to\train_manifest.csv --image-root D:\path\to\competition-data --checkpoint .\output\competition\stage1\checkpoint-best.pth --output .\output\competition\selftrain_pseudo.csv
python train.py --dataset competition --nb_classes 10 --model eva02_small_patch16_xattn_fusedLN_SwiGLU_preln_RoPE --input_size 224 --batch_size 2 --accum_iter 8 --num_workers 2 --epochs 10 --eval_interval 1 --data_path D:\path\to\competition-data --train_list D:\path\to\train_manifest.csv --test_list D:\path\to\val_manifest.csv --finetune D:\path\to\eva_x_small_patch16_merged520k_mim.pt --competition_init_checkpoint .\output\competition\stage1\checkpoint-best.pth --pseudo_label_list .\output\competition\selftrain_pseudo.csv --output_dir .\output\competition\stage2 --log_dir .\output\competition\stage2 --seed 42 --use_mean_pooling
```

Pseudo-labels are soft probabilities for Pneumothorax, Consolidation, and
Pneumonia only; known labels and other classes are retained. Since the first
stage is run on the training Studies themselves, treat this as a baseline and
compare it with the U-Ignore-only run.

The official submission template is intentionally not generated here. Its
column layout and whether it expects one row per class or one row per Study
must be confirmed from the downloaded template before final export.

## Predict

The inference command consumes an unlabeled normalized manifest and a
fine-tuned EVA-X checkpoint. It writes one row per Study with columns
`probability_1` through `probability_10`; this is an internal probability file,
not the official submission format.

```powershell
python -m competition.predict --manifest D:\path\to\test_manifest.csv --image-root D:\path\to\competition-data --checkpoint .\output\competition\seed42\checkpoint-best.pth --output .\output\competition\test_probabilities.csv --batch-size 8 --num-workers 2
```

export PYTHONPATH="/s/mlopezs8/CloneCharter/src:$PYTHONPATH"
accelerate launch --num_processes 1 src/auto_charter/scripts/train.py \
  --dataset dataset/train \
  --streaming \
  --max-shards-in-memory 1 \
  --batch-size 16 --grad-accum 2 --num-epochs 1500 \
  --steps-per-epoch 2000 \
  --output-dir ./checkpoints/run1 --mixed-precision bf16
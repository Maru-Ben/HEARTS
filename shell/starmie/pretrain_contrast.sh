echo "1/3 Training on Santos benchmark"
CUDA_VISIBLE_DEVICES=0 python scripts/starmie/run_pretrain.py \
  --task santos \
  --batch_size 64 \
  --lr 5e-5 \
  --lm roberta \
  --n_epochs 2 \
  --max_len 256 \
  --projector 768 \
  --save_model \
  --table_order column \
  --augment_op drop_col \
  --sample_meth tfidf_entity \
  --fp16 \
  --run_id 0


CUDA_VISIBLE_DEVICES=0 python scripts/starmie/run_pretrain.py \
  --task santos \
  --batch_size 64 \
  --lr 5e-5 \
  --lm roberta \
  --n_epochs 1 \
  --max_len 256 \
  --projector 768 \
  --save_model \
  --table_order column \
  --augment_op shuffle_col \
  --sample_meth tfidf_entity \
  --fp16 \
  --run_id 1

echo "2/3 Training on TUS benchmark"
CUDA_VISIBLE_DEVICES=0 python scripts/starmie/run_pretrain.py \
  --task tus \
  --batch_size 64 \
  --lr 5e-5 \
  --lm roberta \
  --n_epochs 2 \
  --max_len 256 \
  --projector 768 \
  --save_model \
  --table_order column \
  --augment_op drop_cell \
  --sample_meth alphaHead \
  --fp16 \
  --run_id 0

CUDA_VISIBLE_DEVICES=0 python scripts/starmie/run_pretrain.py \
  --task tus \
  --batch_size 64 \
  --lr 5e-5 \
  --lm roberta \
  --n_epochs 2 \
  --max_len 256 \
  --projector 768 \
  --save_model \
  --table_order column \
  --augment_op shuffle_col \
  --sample_meth alphaHead \
  --fp16 \
  --run_id 1

echo "3/3 Training on TUS Large benchmark"
CUDA_VISIBLE_DEVICES=0 python scripts/starmie/run_pretrain.py \
  --task tusLarge \
  --batch_size 64 \
  --lr 5e-5 \
  --lm roberta \
  --n_epochs 2 \
  --max_len 256 \
  --projector 768 \
  --save_model \
  --table_order column \
  --augment_op drop_cell \
  --sample_meth tfidf_entity \
  --fp16 \
  --run_id 0

CUDA_VISIBLE_DEVICES=0 python scripts/starmie/run_pretrain.py \
  --task tusLarge \
  --batch_size 64 \
  --lr 5e-5 \
  --lm roberta \
  --n_epochs 2 \
  --max_len 256 \
  --projector 768 \
  --save_model \
  --table_order column \
  --augment_op shuffle_col \
  --sample_meth tfidf_entity \
  --fp16 \
  --run_id 1
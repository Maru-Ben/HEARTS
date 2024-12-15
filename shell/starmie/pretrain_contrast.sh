echo "1/4 Training Santos"
CUDA_VISIBLE_DEVICES=0 python scripts/starmie/run_pretrain.py \
  --task santos \
  --batch_size 64 \
  --lr 5e-5 \
  --lm roberta \
  --n_epochs 10 \
  --max_len 128 \
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
  --n_epochs 10 \
  --max_len 128 \
  --projector 768 \
  --save_model \
  --table_order column \
  --augment_op shuffle_col \
  --sample_meth tfidf_entity \
  --fp16 \
  --run_id 0

echo "2/4 Training TUS"
CUDA_VISIBLE_DEVICES=0 python scripts/starmie/run_pretrain.py \
  --task tus \
  --batch_size 64 \
  --lr 5e-5 \
  --lm roberta \
  --n_epochs 10 \
  --max_len 128 \
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
  --n_epochs 10 \
  --max_len 128 \
  --projector 768 \
  --save_model \
  --table_order column \
  --augment_op shuffle_col \
  --sample_meth alphaHead \
  --fp16 \
  --run_id 0

echo "3/4 Training TUS Large"
CUDA_VISIBLE_DEVICES=0 python scripts/starmie/run_pretrain.py \
  --task tusLarge \
  --batch_size 64 \
  --lr 5e-5 \
  --lm roberta \
  --n_epochs 10 \
  --max_len 128 \
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
  --n_epochs 10 \
  --max_len 128 \
  --projector 768 \
  --save_model \
  --table_order column \
  --augment_op shuffle_col \
  --sample_meth tfidf_entity \
  --fp16 \
  --run_id 0

echo "4/4 Training Pylon"
CUDA_VISIBLE_DEVICES=0 python scripts/starmie/run_pretrain.py \
  --task pylon \
  --batch_size 64 \
  --lr 5e-5 \
  --lm roberta \
  --n_epochs 10 \
  --max_len 128 \
  --projector 768 \
  --save_model \
  --table_order column \
  --augment_op drop_col \
  --sample_meth tfidf_entity \
  --fp16 \
  --run_id 0

CUDA_VISIBLE_DEVICES=0 python scripts/starmie/run_pretrain.py \
  --task pylon \
  --batch_size 64 \
  --lr 5e-5 \
  --lm roberta \
  --n_epochs 10 \
  --max_len 128 \
  --projector 768 \
  --save_model \
  --table_order column \
  --augment_op shuffle_col \
  --sample_meth tfidf_entity \
  --fp16 \
  --run_id 0



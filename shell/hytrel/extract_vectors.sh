
python scripts/hytrel/extractVectors.py --benchmark santos \
    --checkpoint_dir checkpoints/hytrel/contrast_pretrained/epoch=4-step=32690.ckpt/checkpoint/mp_rank_00_model_states.pt

python scripts/hytrel/extractVectors.py --benchmark tus \
    --checkpoint_dir checkpoints/hytrel/contrast_pretrained/epoch=4-step=32690.ckpt/checkpoint/mp_rank_00_model_states.pt

python scripts/hytrel/extractVectors.py --benchmark tusLarge \
    --checkpoint_dir checkpoints/hytrel/contrast_pretrained/epoch=4-step=32690.ckpt/checkpoint/mp_rank_00_model_states.pt

python scripts/hytrel/extractVectors.py --benchmark pylon \
    --checkpoint_dir checkpoints/hytrel/contrast_pretrained/epoch=4-step=32690.ckpt/checkpoint/mp_rank_00_model_states.pt

    

# python scripts/hytrel/extractVectors.py --benchmark santos \
#     --checkpoint_dir checkpoints/hytrel/santos_contrast_scratch/best.ckpt/checkpoint/mp_rank_00_model_states.pt

# python scripts/hytrel/extractVectors.py --benchmark tus \
#     --checkpoint_dir checkpoints/hytrel/tus_contrast_scratch/best/mp_rank_00_model_states.pt

# python scripts/hytrel/extractVectors.py --benchmark tusLarge \
#     --checkpoint_dir checkpoints/hytrel/tusLarge_contrast_scratch/best/mp_rank_00_model_states.pt

# python scripts/hytrel/extractVectors.py --benchmark pylon \
#     --checkpoint_dir checkpoints/hytrel/pylon_contrast_scratch/best/mp_rank_00_model_states.pt



# python scripts/hytrel/extractVectors.py --benchmark santos \
#     --checkpoint_dir checkpoints/hytrel/santos_contrast_finetuned/best/mp_rank_00_model_states.pt

# python scripts/hytrel/extractVectors.py --benchmark tus \
#     --checkpoint_dir checkpoints/hytrel/tus_contrast_finetuned/best/mp_rank_00_model_states.pt

# python scripts/hytrel/extractVectors.py --benchmark tusLarge \
#     --checkpoint_dir checkpoints/hytrel/tusLarge_contrast_finetuned/best/mp_rank_00_model_states.pt

# python scripts/hytrel/extractVectors.py --benchmark pylon \
#     --checkpoint_dir checkpoints/hytrel/pylon_contrast_finetuned/best/mp_rank_00_model_states.pt


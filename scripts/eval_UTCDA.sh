OUTPUT_DIR='./checkpoints/'
DATA_PATH='/mnt/data2t/datasets/UTCDA/video_test.txt'

MODEL_PATH='./checkpoints/checkpoint-24.pth'
python evaluate_loc.py \
    --model vit_large_patch16_224 \
    --data_set UTCDA \
    --nb_classes 7 \
    --data_path ${DATA_PATH} \
    --finetune ${MODEL_PATH} \
    --log_dir ${OUTPUT_DIR} \
    --output_dir ${OUTPUT_DIR} \
    --csv_file ${DATA_PATH} \
    --batch_size 2 \
    --num_sample 1 \
    --input_size 224 \
    --short_side_size 224\
    --save_ckpt_freq 10 \
    --num_frames 16 \
    --sampling_rate 1 \
    --opt lion \
    --opt_betas 0.9 0.99 \
    --weight_decay 0.05 \
    --epochs 25 \
    --lr 2e-3 \
    --clip_stride 15 \
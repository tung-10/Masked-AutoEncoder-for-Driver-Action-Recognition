DATA_PATH='/mnt/data2t/datasets/UTCDA/videos_splited'
MODEL_PATH='vit_l_hybrid_pt_800e_k700_ft.pth'
OUTPUT_DIR='./checkpoints'
CUDA_VISIBLE_DEVICES=0 python run_class_finetuning.py \
    --model vit_large_patch16_224 \
    --data_set UTCDA \
    --nb_classes 7 \
    --data_path ${DATA_PATH} \
    --finetune ${MODEL_PATH} \
    --log_dir ${OUTPUT_DIR} \
    --output_dir ${OUTPUT_DIR} \
    --batch_size 2 \
    --num_sample 1 \
    --input_size 224 \
    --short_side_size 224\
    --save_ckpt_freq 2 \
    --num_frames 16 \
    --sampling_rate 1 \
    --opt lion \
    --with_checkpoint \
    # --disable_eval_during_finetuning \
    # --eval \
    --num_workers 5 \
    --opt_betas 0.9 0.999 \
    --weight_decay 0.05 \
    --warmup_epochs 5 \
    --epochs 20 \
    --test_num_segment 10 \
    --test_num_crop 1 \
    --lr 1e-3 \
    --layer_decay 0.75 

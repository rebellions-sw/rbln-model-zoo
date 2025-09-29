: ${CHECKPOINT_DIR:="./ckpt"}
: ${RBLN_DIR:="./rbln_ckpt_distil_perf"}
: ${VIDEO_SAVE_DIR:="./outputs/distil_perf"}

python inference.py \
    transfer \
    --checkpoint_dir $CHECKPOINT_DIR \
    --controlnet_specs assets/single_test.json \
    --video_save_folder $VIDEO_SAVE_DIR \
    --rbln_dir $RBLN_DIR \
    --use_distilled \
    --num_steps 1 \
    --upsample_prompt

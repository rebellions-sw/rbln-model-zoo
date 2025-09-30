: ${CHECKPOINT_DIR:="./ckpt"}
: ${RBLN_DIR:="./rbln_ckpt_upscaler_perf"}
: ${VIDEO_SAVE_DIR:="./outputs/upscaler_perf"}

python inference.py \
    transfer \
    --checkpoint_dir $CHECKPOINT_DIR \
    --controlnet_specs assets/upscaler_test.json \
    --video_save_folder $VIDEO_SAVE_DIR \
    --rbln_dir $RBLN_DIR \
    --upsample_prompt \
    --num_steps 10

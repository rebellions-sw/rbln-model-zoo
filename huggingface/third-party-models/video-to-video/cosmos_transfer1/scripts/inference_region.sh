: ${CHECKPOINT_DIR:="./ckpt"}
: ${RBLN_DIR:="./rbln_ckpt_region"}
: ${VIDEO_SAVE_DIR:="./outputs/region"}

python inference.py \
    transfer \
    --checkpoint_dir $CHECKPOINT_DIR \
    --controlnet_specs assets/regional_prompt_test.json \
    --video_save_folder $VIDEO_SAVE_DIR \
    --rbln_dir $RBLN_DIR \
    --upsample_prompt

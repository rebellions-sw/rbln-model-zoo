: ${CHECKPOINT_DIR:="./ckpt"}
: ${RBLN_DIR:="./rbln_ckpt_av_perf"}
: ${VIDEO_SAVE_DIR:="./outputs/av_perf"}

python inference.py \
    transfer \
    --is_av_sample \
    --sigma_max 80 \
    --checkpoint_dir $CHECKPOINT_DIR \
    --controlnet_specs assets/sample_av_multi_control_spec.json \
    --video_save_folder $VIDEO_SAVE_DIR \
    --rbln_dir $RBLN_DIR \
    --upsample_prompt

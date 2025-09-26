: ${CHECKPOINT_DIR:="./ckpt"}
: ${RBLN_DIR:="./rbln_ckpt_av"}
: ${HEIGHT:="704"}
: ${WIDTH:="1280"}

python compile.py \
    transfer \
    --is_av_sample \
    --checkpoint_dir $CHECKPOINT_DIR \
    --controlnet_specs assets/sample_av_multi_control_spec.json \
    --rbln_dir $RBLN_DIR \
    --upsample_prompt \
    --height $HEIGHT \
    --width $WIDTH

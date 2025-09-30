: ${CHECKPOINT_DIR:="./ckpt"}
: ${RBLN_DIR:="./rbln_ckpt_upscaler"}
: ${HEIGHT:="704"}
: ${WIDTH:="1280"}

python compile.py \
    transfer \
    --checkpoint_dir $CHECKPOINT_DIR \
    --controlnet_specs assets/upscaler_test.json \
    --rbln_dir $RBLN_DIR \
    --upsample_prompt \
    --height $HEIGHT \
    --width $WIDTH

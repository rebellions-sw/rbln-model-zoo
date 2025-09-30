: ${CHECKPOINT_DIR:="./ckpt"}
: ${RBLN_DIR:="./rbln_ckpt_perf"}
: ${HEIGHT:="704"}
: ${WIDTH:="1280"}

python compile.py \
    transfer \
    --checkpoint_dir $CHECKPOINT_DIR \
    --controlnet_specs assets/single_test.json \
    --rbln_dir $RBLN_DIR \
    --upsample_prompt \
    --use_perf \
    --height $HEIGHT \
    --width $WIDTH

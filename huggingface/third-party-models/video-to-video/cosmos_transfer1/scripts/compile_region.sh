: ${CHECKPOINT_DIR:="./ckpt"}
: ${RBLN_DIR:="./rbln_ckpt_region"}
: ${HEIGHT:="704"}
: ${WIDTH:="1280"}

python compile.py \
    transfer \
    --checkpoint_dir $CHECKPOINT_DIR \
    --controlnet_specs assets/regional_prompt_test.json \
    --rbln_dir $RBLN_DIR \
    --use_regional_prompts \
    --num_regions 3 \
    --upsample_prompt \
    --height $HEIGHT \
    --width $WIDTH

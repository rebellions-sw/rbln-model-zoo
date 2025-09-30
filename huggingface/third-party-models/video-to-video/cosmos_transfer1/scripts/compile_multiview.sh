: ${CHECKPOINT_DIR:="./ckpt"}
: ${RBLN_DIR:="./rbln_ckpt_multiview"}

python compile.py \
    transfer_multiview \
    --checkpoint_dir $CHECKPOINT_DIR \
    --controlnet_specs assets/sample_av_hdmap_multiview_spec.json \
    --rbln_dir $RBLN_DIR

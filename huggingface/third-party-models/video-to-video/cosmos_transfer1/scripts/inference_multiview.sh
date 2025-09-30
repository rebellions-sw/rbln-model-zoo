: ${CHECKPOINT_DIR:="./ckpt"}
: ${RBLN_DIR:="./rbln_ckpt_multiview"}
: ${VIDEO_SAVE_DIR:="./outputs/multiview"}

python inference.py \
    transfer_multiview \
    --prompt "The video is captured from a camera mounted on a car. The camera is facing forward. The video captures a driving scene on a multi-lane highway during the day. The sky is clear and blue, indicating good weather conditions. The road is relatively busy with several cars and trucks in motion. A red sedan is driving in the left lane, followed by a black pickup truck in the right lane. The vehicles are maintaining a safe distance from each other. On the right side of the road, there are speed limit signs indicating a limit of 65 mph. The surrounding area includes a mix of greenery and industrial buildings, with hills visible in the distance. The overall environment appears to be a typical day on a highway with moderate traffic. The golden light of the late afternoon bathes the highway, casting long shadows and creating a warm, serene atmosphere. The sky is a mix of orange and blue, with the sun low on the horizon. The red sedan in the left lane reflects the golden hues, while the black pickup truck in the right lane casts a distinct shadow on the pavement. The speed limit signs stand out clearly under the fading sunlight. The surrounding greenery glows with a rich, warm tone, and the industrial buildings take on a softened appearance in the sunset." \
    --checkpoint_dir $CHECKPOINT_DIR \
    --view_condition_video assets/sample_av_mv_input_rgb.mp4 \
    --controlnet_specs assets/sample_av_hdmap_multiview_spec.json \
    --video_save_folder $VIDEO_SAVE_DIR \
    --rbln_dir $RBLN_DIR \
    --guidance 3 \
    --num_steps 30

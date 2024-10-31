ckpts_path=(
)

for ckpt_path in ${ckpts_path[@]}; do
  echo "🎃$ckpt_path"
  # Launch LIBERO-Object evals
  python internvla/run_libero_eval.py \
    --model_family openvla \
    --pretrained_checkpoint $ckpt_path \
    --task_suite_name libero_object \
    --center_crop True
done

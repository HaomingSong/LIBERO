ckpts_path=(
  ../pretrained/2024-10-12_23-43-40_libero_spatial_no_noops_libero_ft_baseline_decay0_0_warmup0_005_linear_lr2e-5_bs32_ga1_node1_gpu4_checkpoint-37260
)

for ckpt_path in ${ckpts_path[@]}; do
  echo "🎃$ckpt_path"
  # Launch LIBERO-Spatial evals
  python internvla/run_libero_eval.py \
    --model_family openvla \
    --pretrained_checkpoint $ckpt_path \
    --task_suite_name libero_spatial \
    --center_crop True
done

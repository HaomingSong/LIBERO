ckpts_path=(
  # pretrained/2024-12-11_00_libero_spatial_no_noops_paligemma_3b_vis_zoe_flash_obs14_spatial_untie_gaussN8194_unicam_lr2e-5_bs32_ga1_node1_gpu4_checkpoint-20700
  # pretrained/2024-12-11_22_libero_spatial_no_noops_paligemma_3b_vis_zoe_flash_obs14_spatial_untie_gaussN8194_unicam_lr4e-5_bs32_ga1_node1_gpu8_checkpoint-10350

 # pretrained/2023-12-12_15_libero_spatial_no_noops_paligemma_3b_vis_zoe_flash_obs14_spatial_untie_gaussN8194_unicam_lr4e-5_bs32_ga1_node1_gpu8_r32_checkpoint-10350
 pretrained/2024-12-12_22_libero_spatial_no_noops_paligemma_3b_vis_zoe_flash_obs14_spatial_untie_gaussN8194_unicam_lr5e-4_bs32_ga1_node1_gpu8_r32_checkpoint-10000
 pretrained/2024-12-12_22_libero_spatial_no_noops_paligemma_3b_vis_zoe_flash_obs14_spatial_untie_gaussN8194_unicam_lr5e-4_bs32_ga1_node1_gpu8_r32_checkpoint-20000
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

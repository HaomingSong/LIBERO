ckpts_path=(
  # spatial
  # -/cpfs01/shared/optimal/vla_ptm/SpatialVLA-PaliGemma-Zoe/outputs/spatialvla_v1_paligemma2_3b_finetune/2024-12-31/18_libero_spatial_no_noops_simplerenv_zoe_N8194_uniform_gpu8_checkpoint120000_lr5e-4_bs32_node1_gpu4_r32_a32_ep200_all-linear+emb_oxe_smp/checkpoint-60000
  # /cpfs01/shared/optimal/vla_ptm/SpatialVLA-PaliGemma-Zoe/outputs/spatialvla_v1_paligemma2_3b_finetune/2024-12-31/18_libero_spatial_no_noops_simplerenv_zoe_N8194_uniform_gpu8_checkpoint120000_lr5e-4_bs32_node1_gpu4_r32_a32_ep200_all-linear+emb_oxe_smp/checkpoint-70000
  # /cpfs01/shared/optimal/vla_ptm/SpatialVLA-PaliGemma-Zoe/outputs/spatialvla_v1_paligemma2_3b_finetune/2024-12-31/18_libero_spatial_no_noops_simplerenv_zoe_N8194_uniform_gpu8_checkpoint120000_lr5e-4_bs32_node1_gpu4_r32_a32_ep200_all-linear+emb_oxe_smp/checkpoint-82800
  # /cpfs01/shared/optimal/vla_ptm/SpatialVLA-PaliGemma-Zoe/outputs/spatialvla_v1_paligemma2_3b_finetune/2025-01-03/13_libero_spatial_no_noops_simplerenv_zoe_N8194_uniform_gpu8_checkpoint120000_lr5e-4_bs32_node1_gpu4_r32_a32_ep200_all-linear+emb_oxe_smp/checkpoint-50000
  # /cpfs01/shared/optimal/vla_ptm/SpatialVLA-PaliGemma-Zoe/outputs/spatialvla_v1_paligemma2_3b_finetune/2025-01-03/13_libero_spatial_no_noops_simplerenv_zoe_N8194_uniform_gpu8_checkpoint120000_lr5e-4_bs32_node1_gpu4_r32_a32_ep200_all-linear+emb_oxe_smp/checkpoint-60000
  # /cpfs01/shared/optimal/vla_ptm/SpatialVLA-PaliGemma-Zoe/outputs/spatialvla_v1_paligemma2_3b_finetune/2025-01-03/13_libero_spatial_no_noops_simplerenv_zoe_N8194_uniform_gpu8_checkpoint120000_lr5e-4_bs32_node1_gpu4_r32_a32_ep200_all-linear+emb_oxe_smp/checkpoint-70000
  # object
  # /cpfs01/shared/optimal/vla_ptm/SpatialVLA-PaliGemma-Zoe/outputs/spatialvla_v1_paligemma2_3b_finetune/2025-01-01/16_libero_object_no_noops_simplerenv_zoe_N8194_uniform_gpu8_checkpoint120000_lr5e-4_bs32_node1_gpu4_r32_a32_ep150_all-linear+emb_oxe_smp_gs/checkpoint-50000
  # /cpfs01/shared/optimal/vla_ptm/SpatialVLA-PaliGemma-Zoe/outputs/spatialvla_v1_paligemma2_3b_finetune/2025-01-01/16_libero_object_no_noops_simplerenv_zoe_N8194_uniform_gpu8_checkpoint120000_lr5e-4_bs32_node1_gpu4_r32_a32_ep150_all-linear+emb_oxe_smp_gs/checkpoint-60000  
  # /cpfs01/shared/optimal/vla_ptm/SpatialVLA-PaliGemma-Zoe/outputs/spatialvla_v1_paligemma2_3b_finetune/2024-12-31/18_libero_object_no_noops_simplerenv_zoe_N8194_uniform_gpu8_checkpoint120000_lr5e-4_bs32_node1_gpu4_r32_a32_ep200_all-linear+emb_oxe_smp/checkpoint-90000
  # /cpfs01/shared/optimal/vla_ptm/SpatialVLA-PaliGemma-Zoe/outputs/spatialvla_v1_paligemma2_3b_finetune/2025-01-01/16_libero_object_no_noops_simplerenv_zoe_N8194_uniform_gpu8_checkpoint120000_lr5e-4_bs32_node1_gpu4_r32_a32_ep150_all-linear+emb_oxe_smp_gs/checkpoint-60000
  # /cpfs01/shared/optimal/vla_ptm/SpatialVLA-PaliGemma-Zoe/outputs/spatialvla_v1_paligemma2_3b_finetune/2025-01-02/14_libero_object_no_noops_simplerenv_zoe_N8194_uniform_gpu8_checkpoint120000_lr5e-4_bs32_node1_gpu4_r32_a32_ep150_all-linear+emb_oxe_obj_smp/checkpoint-78600
  # /cpfs01/shared/optimal/vla_ptm/SpatialVLA-PaliGemma-Zoe/outputs/spatialvla_v1_paligemma2_3b_finetune/2025-01-04/22_libero_object_no_noops_simpler_env2_zoe_uniform8194_checkpoint120000_lr5e-4_bs32_node1_gpu4_r32_a32_ep150_all-linear+emb_smp2_sigma0/checkpoint-70000
  # /cpfs01/shared/optimal/vla_ptm/SpatialVLA-PaliGemma-Zoe/outputs/spatialvla_v1_paligemma2_3b_finetune/2025-01-04/22_libero_object_no_noops_simpler_env2_zoe_uniform8194_checkpoint120000_lr5e-4_bs32_node1_gpu4_r32_a32_ep150_all-linear+emb_smp2_sigma0/checkpoint-60000
  # /cpfs01/shared/optimal/vla_ptm/SpatialVLA-PaliGemma-Zoe/outputs/spatialvla_v1_paligemma2_3b_finetune/2025-01-04/22_libero_object_no_noops_simpler_env2_zoe_uniform8194_checkpoint120000_lr5e-4_bs32_node1_gpu4_r32_a32_ep150_all-linear+emb_smp2_sigma0/checkpoint-50000
  # /cpfs01/shared/optimal/vla_ptm/SpatialVLA-PaliGemma-Zoe/outputs/spatialvla_v1_paligemma2_3b_finetune/2025-01-01/16_libero_object_no_noops_simplerenv_zoe_N8194_uniform_gpu8_checkpoint120000_lr5e-4_bs32_node1_gpu4_r32_a32_ep150_all-linear+emb_oxe_smp_gs/checkpoint-78600
  # /cpfs01/shared/optimal/vla_ptm/SpatialVLA-PaliGemma-Zoe/outputs/spatialvla_v1_paligemma2_3b_finetune/2025-01-01/16_libero_object_no_noops_simplerenv_zoe_N8194_uniform_gpu8_checkpoint120000_lr5e-4_bs32_node1_gpu4_r32_a32_ep150_all-linear+emb_oxe_smp_gs/checkpoint-70000

  # goal

  # 10

  # ablations
  # /cpfs01/shared/optimal/vla_ptm/SpatialVLA-PaliGemma-Zoe/outputs/spatialvla_paligemma2_3b_ablation_sft/2025-01-04/23_libero_spatial_no_noops_simplerenv_zoe_N8194_uniform_gpu8_checkpoint120000_lr5e-4_bs32_node1_gpu4_r32_a32_ep150_all-linear_ab_smp/checkpoint-50000
  # /cpfs01/shared/optimal/vla_ptm/SpatialVLA-PaliGemma-Zoe/outputs/spatialvla_paligemma2_3b_ablation_sft/2025-01-05/00_libero_object_no_noops_simplerenv_zoe_N8194_uniform_gpu8_checkpoint120000_lr5e-4_bs32_node1_gpu4_r32_a32_ep150_all-linear_ab_smp/checkpoint-50000

  # /cpfs01/shared/optimal/vla_ptm/SpatialVLA-PaliGemma-Zoe/outputs/spatialvla_paligemma2_3b_ablation_sft/2025-01-04/23_libero_spatial_no_noops_simplerenv_zoe_N8194_uniform_gpu8_checkpoint120000_lr5e-4_bs32_node1_gpu4_r32_a32_ep150_all-linear_ab_smp/checkpoint-60000
  # /cpfs01/shared/optimal/vla_ptm/SpatialVLA-PaliGemma-Zoe/outputs/spatialvla_paligemma2_3b_ablation_sft/2025-01-05/00_libero_object_no_noops_simplerenv_zoe_N8194_uniform_gpu8_checkpoint120000_lr5e-4_bs32_node1_gpu4_r32_a32_ep150_all-linear_ab_smp/checkpoint-60000
)

# ensembler=vanilla
ensembler=adpt

for i in ${!ckpts_path[@]}; do
  ckpt_path=${ckpts_path[$i]}
  echo "🎃$ckpt_path"
  # Launch LIBERO-Spatial evals
  CUDA_VISIBLE_DEVICES=1 python internvla/run_libero_eval.py \
    --model_family openvla \
    --pretrained_checkpoint $ckpt_path \
    --task_suite_name auto \
    --num_trials_per_task 50 \
    --run_id_note $(basename $(dirname $ckpt_path))_$(basename $ckpt_path)_${ensembler} \
    --ensembler ${ensembler} \
    --center_crop True \
    --local_log_dir sota
  # cp -r $ckpt_path /oss/vla_ptm_hwfile/LIBERO_CKPT/
done
ckpts_path=(
  # demo
  # /cpfs01/shared/optimal/vla_ptm/SpatialVLA-PaliGemma-Zoe/outputs/spatialvla_v1_paligemma2_3b_finetune/2025-01-03/13_libero_spatial_no_noops_simplerenv_zoe_N8194_uniform_gpu8_checkpoint120000_lr5e-4_bs32_node1_gpu4_r32_a32_ep200_all-linear+emb_oxe_smp/checkpoint-60000
  # /cpfs01/shared/optimal/vla_ptm/SpatialVLA-PaliGemma-Zoe/outputs/spatialvla_v1_paligemma2_3b_finetune/2025-01-01/16_libero_object_no_noops_simplerenv_zoe_N8194_uniform_gpu8_checkpoint120000_lr5e-4_bs32_node1_gpu4_r32_a32_ep150_all-linear+emb_oxe_smp_gs/checkpoint-60000
  # /cpfs01/shared/optimal/vla_ptm/SpatialVLA-PaliGemma-Zoe/outputs/spatialvla_v1_paligemma2_3b_finetune/2025-01-10/15_libero_goal_no_noops_2025-01-05_09-12-37_oxe_spatial_vla_paligemma3b_zoe_gsN8194_gpu64-120k_lr2e-5_bs32_node1_gpu4_r0_a0_ep200_none_oxe_adpt_fea_sft/checkpoint-30000
  /cpfs01/shared/optimal/vla_ptm/SpatialVLA-PaliGemma-Zoe/outputs/spatialvla_v1_paligemma2_3b_finetune/2025-01-02/20_libero_10_no_noops_simplerenv_zoe_N8194_uniform_gpu8_checkpoint120000_lr5e-4_bs32_node1_gpu4_r32_a32_ep150_all-linear+emb_oxe_obj_smp/checkpoint-70000
)

# ensembler=vanilla
ensembler=adpt

for i in ${!ckpts_path[@]}; do
  ckpt_path=${ckpts_path[$i]}
  echo "🎃$ckpt_path"
  # Launch LIBERO-Spatial evals
  CUDA_VISIBLE_DEVICES=3 python internvla/run_libero_eval.py \
    --model_family openvla \
    --pretrained_checkpoint $ckpt_path \
    --task_suite_name auto \
    --num_trials_per_task 10 \
    --run_id_note $(basename $(dirname $ckpt_path))_$(basename $ckpt_path)_${ensembler} \
    --ensembler ${ensembler} \
    --local_log_dir demo
done
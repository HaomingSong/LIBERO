tasks=(
  # eval_goal.sh
  # eval_long.sh
  internvla/scripts/eval_object.sh
  internvla/scripts/eval_spatial.sh
)

for task in ${tasks[@]}; do
  echo "🎃$task"
  bash $task
done


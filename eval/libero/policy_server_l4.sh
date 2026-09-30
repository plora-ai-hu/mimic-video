#!/usr/bin/env bash
# Serve the mimic-video LIBERO policy on one 24 GB GPU (L4) for viewer_client.py.
#
#   ./policy_server_l4.sh                  # libero_spatial_full on 127.0.0.1:8766
#   ./policy_server_l4.sh object_full      # another downloaded checkpoint
#
# Needs libero_prompt_embeddings.pt from precompute_prompt_embeddings.py (T5-11B does not
# fit next to the model). /usr/local/cuda* is dropped from LD_LIBRARY_PATH because the
# host's CUDA 13.2 ships cuDNN 9.20, and loading its libcudnn_graph alongside the venv's
# cuDNN 9.5 makes the video tokenizer's conv3d fail with "ptrDesc->finalize()".
set -euo pipefail
cd "$(dirname "$0")"

model=${1:-spatial_full}
suite=libero_${model%_*}
ck=../../model/checkpoints
action_model=$(ls $ck/action_decoder/w2a_libero_${model}_*.pt)

source ../../model/.venv/bin/activate
LD_LIBRARY_PATH=$(echo "${LD_LIBRARY_PATH:-}" | tr ':' '\n' | grep -v "/usr/local/cuda" | paste -sd:)
export LD_LIBRARY_PATH PYTHONPATH=LIBERO TOKENIZERS_PARALLELISM=false MUJOCO_GL=egl

exec python policy_server.py \
  --vam_experiment_name "$(basename "$action_model" | sed 's/_iter_[0-9]*\.pt$//')" \
  --vam_video_model_path "$(ls $ck/video_backbone/v2w_${suite}_*.pt)" \
  --vam_action_model_path "$action_model" \
  --vam_dataset_statistics_path "$ck/dataset_statistics/libero_${model}.json" \
  --vam_prompt_embeddings_path libero_prompt_embeddings.pt \
  --task_suite_name "$suite" \
  --port "${PORT:-8766}"

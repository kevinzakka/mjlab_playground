"""Register KUKA+SHARPA dexterous tool task."""

from mjlab.tasks.registry import register_mjlab_task

from mjlab_playground.dexterous_tool.config.kuka_sharpa.env_cfgs import (
  kuka_sharpa_dexterous_tool_env_cfg,
)
from mjlab_playground.dexterous_tool.config.kuka_sharpa.rl_cfg import (
  kuka_sharpa_dexterous_tool_ppo_runner_cfg,
)

register_mjlab_task(
  task_id="Mjlab-DexterousTool-KukaSharpa",
  env_cfg=kuka_sharpa_dexterous_tool_env_cfg(),
  play_env_cfg=kuka_sharpa_dexterous_tool_env_cfg(play=True),
  rl_cfg=kuka_sharpa_dexterous_tool_ppo_runner_cfg(),
)

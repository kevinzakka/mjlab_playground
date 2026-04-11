"""Register floating SHARPA dexterous tool task."""

from mjlab.tasks.registry import register_mjlab_task

from mjlab_playground.dexterous_tool.config.floating_sharpa.env_cfgs import (
  floating_sharpa_dexterous_tool_env_cfg,
)
from mjlab_playground.dexterous_tool.config.floating_sharpa.rl_cfg import (
  floating_sharpa_dexterous_tool_ppo_runner_cfg,
)

register_mjlab_task(
  task_id="Mjlab-DexterousTool-FloatingSharpa",
  env_cfg=floating_sharpa_dexterous_tool_env_cfg(),
  play_env_cfg=floating_sharpa_dexterous_tool_env_cfg(play=True),
  rl_cfg=floating_sharpa_dexterous_tool_ppo_runner_cfg(),
)

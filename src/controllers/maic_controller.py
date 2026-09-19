"""MAIC multi-agent controller.

Builds on BasicMAC so observation-delay handling and makers stay consistent
with the rest of this repo. The extra return dict carries MAIC auxiliary
losses to the learner without changing run/main.
"""

from __future__ import annotations

import torch as th

from controllers.basic_controller import BasicMAC


class MAICMAC(BasicMAC):
    """Parameter-sharing controller for MAIC."""

    def select_actions(self, ep_batch, t_ep, t_env, bs=slice(None), test_mode=False):
        avail_actions = ep_batch["avail_actions"][:, t_ep]
        agent_outputs, _ = self.forward(
            ep_batch, t_ep, test_mode=test_mode, train_mode=False
        )
        chosen_actions = self.action_selector.select_action(
            agent_outputs[bs], avail_actions[bs], t_env, test_mode=test_mode
        )
        return chosen_actions

    def forward(self, ep_batch, t, test_mode=False, **kwargs):
        agent_inputs = self._build_inputs(ep_batch, t)
        avail_actions = ep_batch["avail_actions"][:, t]

        if test_mode:
            self.agent.eval()
        else:
            self.agent.train()

        agent_outs, self.hidden_states, losses = self.agent.forward(
            agent_inputs,
            self.hidden_states,
            ep_batch.batch_size,
            test_mode=test_mode,
            **kwargs,
        )

        if self.agent_output_type == "pi_logits":
            if getattr(self.args, "mask_before_softmax", True):
                reshaped_avail_actions = avail_actions.reshape(
                    ep_batch.batch_size * self.n_agents, -1
                )
                agent_outs[reshaped_avail_actions == 0] = -1e10
            agent_outs = th.nn.functional.softmax(agent_outs, dim=-1)

        return agent_outs.view(ep_batch.batch_size, self.n_agents, -1), losses

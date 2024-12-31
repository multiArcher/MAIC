from .entity_controller import EntityMAC


# This multi-agent controller shares parameters between agents
class EntityMAICMAC(EntityMAC):
    def __init__(self, scheme, groups, args):
        super(EntityMAICMAC, self).__init__(scheme, groups, args)

    def select_actions(self, ep_batch, t_ep, t_env, bs=slice(None), test_mode=False):
        # Only select actions for the selected batch elements in bs
        avail_actions = ep_batch["avail_actions"][:, t_ep]
        agent_outputs, _ = self.forward(ep_batch, t_ep, test_mode=test_mode, train_mode=False)
        chosen_actions = self.action_selector.select_action(agent_outputs[bs], avail_actions[bs], t_env, test_mode=test_mode)
        return chosen_actions

    def forward(self, ep_batch, t, test_mode=False, **kwargs):
        if int_t:= isinstance(t, int):
            t = slice(t, t + 1)

        agent_inputs = self._build_inputs(ep_batch, t)
        avail_actions = ep_batch["avail_actions"][:, t]
        agent_outs, self.hidden_states, losses = self.agent.forward(agent_inputs, self.hidden_states, ep_batch.batch_size,
            test_mode=test_mode, **kwargs)

        return agent_outs.view(ep_batch.batch_size, self.n_agents, -1), losses

    def init_hidden(self, batch_size):
        self.hidden_states = self.agent.init_hidden().unsqueeze(0).expand(batch_size, self.n_agents, -1)  # bav

from typing import Any

import torch

from components.observation_delay_model import ObservationDelayModel
from controllers.mac import MAC
from modules.rdc import RDCEchoPredictor
from utils.maker import ActionSelectorMaker, AgentMaker


class RDCMAC(MAC):
    """QMIX controller with full-observation Echo delay compensation."""

    def __init__(self, scheme: dict, groups: dict, args):
        super().__init__(scheme, groups, args)
        self.args = args
        self.device = args.device
        self.n_agents = args.n_agents
        self.n_actions = args.n_actions
        self.obs_dim = int(scheme["obs"]["vshape"])
        self.agent_output_type = args.agent_output_type

        self._build_agents(self._get_input_shape(scheme))
        self.predictor = RDCEchoPredictor(self.obs_dim, self.n_actions, args)
        self.action_selector = ActionSelectorMaker.make(args.action_selector, args)
        self.observation_delay_model = ObservationDelayModel(args)

        self.hidden_states = None
        self._delayed_obs_cache = None
        self._delay_cache = None
        self._source_time_cache = None
        self._cached_steps = None

    def select_actions(
        self,
        ep_batch,
        t_ep: int,
        t_env: int,
        bs=slice(None),
        test_mode: bool = False,
    ) -> Any:
        if test_mode:
            self.agent.eval()
        else:
            self.agent.train()
        self.predictor.eval()
        avail_actions = ep_batch["avail_actions"][:, t_ep]
        q_values = self.forward(ep_batch, t_ep, test_mode=test_mode)
        return self.action_selector.select_action(
            q_values[bs], avail_actions[bs], t_env, test_mode=test_mode
        )

    def forward(
        self,
        ep_batch,
        t,
        test_mode: bool = False,
        predicted_obs: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        if predicted_obs is None:
            if not isinstance(t, int):
                raise TypeError("RDC online prediction expects an integer timestep")
            predicted_obs = self._predict_online(ep_batch, t, test_mode)

        agent_inputs = self._build_inputs(ep_batch, t, predicted_obs)
        q_values, self.hidden_states = self.agent(agent_inputs, self.hidden_states)
        return q_values.view(ep_batch.batch_size, self.n_agents, -1)

    def predict_batch(
        self,
        ep_batch,
        *,
        training: bool,
        compute_loss: bool,
        query_indices: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        time_slice = slice(0, ep_batch.max_seq_length)
        delayed_obs, delays, source_times = self.observation_delay_model.apply_with_metadata(
            ep_batch["obs"], time_slice, training=training
        )
        valid_mask = ep_batch["filled"].float() if compute_loss else None
        output = self.predictor(
            delayed_obs,
            delays,
            ep_batch["actions_onehot"],
            query_indices=query_indices,
            clean_obs=ep_batch["obs"] if compute_loss else None,
            valid_mask=valid_mask,
        )
        output.update(
            {
                "delayed_obs": delayed_obs,
                "delays": delays,
                "source_times": source_times,
            }
        )
        return output

    def _predict_online(self, ep_batch, t: int, test_mode: bool) -> torch.Tensor:
        self._ensure_delay_cache(ep_batch)
        if not bool(self._cached_steps[t]):
            delayed, delays, source_times = self.observation_delay_model.apply_with_metadata(
                ep_batch["obs"], slice(t, t + 1), training=not test_mode
            )
            self._delayed_obs_cache[:, t : t + 1] = delayed
            self._delay_cache[:, t : t + 1] = delays
            self._source_time_cache[:, t : t + 1] = source_times
            self._cached_steps[t] = True

        history = slice(0, t + 1)
        query = torch.tensor([t], device=ep_batch["obs"].device, dtype=torch.long)
        output = self.predictor(
            self._delayed_obs_cache[:, history],
            self._delay_cache[:, history],
            ep_batch["actions_onehot"][:, history],
            query_indices=query,
        )
        return output["predicted_obs"]

    def _ensure_delay_cache(self, ep_batch) -> None:
        expected_shape = ep_batch["obs"].shape
        if self._delayed_obs_cache is not None and self._delayed_obs_cache.shape == expected_shape:
            return
        self._delayed_obs_cache = torch.zeros_like(ep_batch["obs"])
        self._delay_cache = torch.zeros(expected_shape[:-1], dtype=torch.long, device=ep_batch.device)
        self._source_time_cache = torch.zeros_like(self._delay_cache)
        self._cached_steps = torch.zeros(expected_shape[1], dtype=torch.bool, device=ep_batch.device)

    def init_hidden(self, batch_size):
        hidden = self.agent.init_hidden()
        self.hidden_states = hidden.unsqueeze(0).expand(batch_size, self.n_agents, -1)
        self._delayed_obs_cache = None
        self._delay_cache = None
        self._source_time_cache = None
        self._cached_steps = None

    def load_state(self, other_mac):
        self.agent.load_state_dict(other_mac.agent.state_dict())

    def save_models(self, path):
        torch.save(self.agent.state_dict(), f"{path}/agent.th")

    def load_models(self, path):
        self.agent.load_state_dict(
            torch.load(f"{path}/agent.th", map_location=lambda storage, loc: storage)
        )

    def _build_agents(self, input_shape):
        self.agent = AgentMaker.make(self.args.agent, input_shape, self.args)

    def _build_inputs(self, batch, t, predicted_obs=None):
        if predicted_obs is None:
            raise ValueError("RDC agent inputs require predicted_obs")
        batch_size = batch.batch_size
        if predicted_obs.dim() == 4:
            obs_t = predicted_obs[:, t] if predicted_obs.size(1) > 1 else predicted_obs[:, 0]
        else:
            obs_t = predicted_obs

        inputs = [obs_t]
        if self.args.obs_last_action:
            if t == 0:
                inputs.append(torch.zeros_like(batch["actions_onehot"][:, t]))
            else:
                inputs.append(batch["actions_onehot"][:, t - 1])
        if self.args.obs_agent_id:
            inputs.append(
                torch.eye(self.n_agents, device=batch.device).unsqueeze(0).expand(
                    batch_size, -1, -1
                )
            )
        return torch.cat(
            [value.reshape(batch_size, self.n_agents, -1) for value in inputs], dim=-1
        )

    def _get_input_shape(self, scheme):
        input_shape = int(scheme["obs"]["vshape"])
        if self.args.obs_last_action:
            input_shape += int(scheme["actions_onehot"]["vshape"][0])
        if self.args.obs_agent_id:
            input_shape += self.n_agents
        return input_shape

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class RDCObservationLayout:
    obs_dim: int
    binary_indices: tuple[int, ...]
    continuous_indices: tuple[int, ...]

    @classmethod
    def from_env_info(cls, env_info: dict) -> "RDCObservationLayout":
        if env_info.get("obs_terrain_height", False):
            raise ValueError("RDC full-observation prediction does not support obs_terrain_height=True")
        if env_info.get("env_obs_last_action", False):
            raise ValueError("RDC expects env_args.obs_last_action=False; MAC action history is supplied separately")
        if env_info.get("obs_timestep_number", False):
            raise ValueError("RDC expects env_args.obs_timestep_number=False")

        move_dim, enemy_shape, ally_shape, own_dim = env_info["obs_components"]
        n_enemies, enemy_dim = enemy_shape
        n_allies, ally_dim = ally_shape
        type_bits = int(env_info["unit_type_bits"])

        binary = set(range(int(move_dim)))
        cursor = int(move_dim)
        for _ in range(int(n_enemies)):
            binary.add(cursor)
            binary.update(range(cursor + enemy_dim - type_bits, cursor + enemy_dim))
            cursor += enemy_dim
        for _ in range(int(n_allies)):
            binary.add(cursor)
            binary.update(range(cursor + ally_dim - type_bits, cursor + ally_dim))
            cursor += ally_dim
        binary.update(range(cursor + own_dim - type_bits, cursor + own_dim))
        cursor += own_dim

        obs_dim = int(env_info["obs_shape"])
        if cursor != obs_dim:
            raise ValueError(f"SMAC observation components sum to {cursor}, expected {obs_dim}")
        binary_indices = tuple(sorted(binary))
        continuous_indices = tuple(i for i in range(obs_dim) if i not in binary)
        return cls(obs_dim, binary_indices, continuous_indices)


class PositionalEncoding(nn.Module):
    def __init__(self, hidden_dim: int, max_len: int):
        super().__init__()
        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, hidden_dim, 2, dtype=torch.float32)
            * (-math.log(10000.0) / hidden_dim)
        )
        encoding = torch.zeros(max_len, hidden_dim)
        encoding[:, 0::2] = torch.sin(position * div_term)
        encoding[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("encoding", encoding.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.encoding[:, : x.size(1)]


class GRUEchoBackbone(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        self.gru = nn.GRUCell(hidden_dim, hidden_dim)

    def encode(self, context: torch.Tensor) -> torch.Tensor:
        batch, queries, agents, history, input_dim = context.shape
        sequence = context.reshape(batch * queries * agents, history, input_dim)
        hidden = sequence.new_zeros(batch * queries * agents, self.hidden_dim)
        for step in range(history):
            embedded = F.relu(self.input_projection(sequence[:, step]), inplace=False)
            hidden = self.gru(embedded, hidden)
        return hidden.view(batch, queries, agents, self.hidden_dim)

    def step(self, inputs: torch.Tensor, hidden: torch.Tensor) -> torch.Tensor:
        shape = inputs.shape[:-1]
        embedded = F.relu(
            self.input_projection(inputs.reshape(-1, inputs.size(-1))),
            inplace=False,
        )
        hidden = self.gru(embedded, hidden.reshape(-1, self.hidden_dim))
        return hidden.view(*shape, self.hidden_dim)


class TransformerEchoBackbone(nn.Module):
    def __init__(
        self,
        input_dim: int,
        obs_dim: int,
        hidden_dim: int,
        max_sequence_len: int,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        self.query_projection = nn.Linear(obs_dim, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=4,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            batch_first=True,
        )
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=4,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=1)
        self.context_position = PositionalEncoding(hidden_dim, max_sequence_len)
        self.query_position = PositionalEncoding(hidden_dim, max_sequence_len)

    def forward(
        self,
        context: torch.Tensor,
        target_sequence: torch.Tensor,
    ) -> torch.Tensor:
        batch, queries, agents, history, input_dim = context.shape
        sequence = context.reshape(batch * queries * agents, history, input_dim)
        memory = self.encoder(self.context_position(F.relu(self.input_projection(sequence), inplace=False)))
        target_len = target_sequence.size(3)
        target = target_sequence.reshape(batch * queries * agents, target_len, -1)
        target = self.query_position(F.relu(self.query_projection(target), inplace=False))
        causal_mask = nn.Transformer.generate_square_subsequent_mask(
            target_len, device=target.device
        )
        hidden = self.decoder(target, memory, tgt_mask=causal_mask)[:, -1]
        return hidden.view(batch, queries, agents, self.hidden_dim)


class RDCEchoPredictor(nn.Module):
    """Echo predictor adapted to EPyMARL's whole-observation delay model."""

    def __init__(self, obs_dim: int, n_actions: int, args):
        super().__init__()
        self.args = args
        self.max_delay = int(args.obs_delay_max)
        self.history_len = int(args.n_expand_action)
        if self.max_delay < 0 or self.history_len < 1:
            raise ValueError("obs_delay_max must be non-negative and n_expand_action must be positive")

        self.layout = RDCObservationLayout.from_env_info(args.env_info)
        if self.layout.obs_dim != obs_dim:
            raise ValueError(f"Predictor obs dim {self.layout.obs_dim} does not match scheme {obs_dim}")

        self.register_buffer(
            "binary_indices",
            torch.tensor(self.layout.binary_indices, dtype=torch.long),
            persistent=False,
        )
        self.register_buffer(
            "continuous_indices",
            torch.tensor(self.layout.continuous_indices, dtype=torch.long),
            persistent=False,
        )

        input_dim = obs_dim + n_actions + 1
        if args.predictor_model == "gru":
            self.backbone = GRUEchoBackbone(input_dim, args.pd_hidden_dim)
        elif args.predictor_model == "transformer":
            if args.transformer_structure != "encoder-decoder":
                raise ValueError("RDC Transformer requires transformer_structure='encoder-decoder'")
            self.backbone = TransformerEchoBackbone(
                input_dim,
                obs_dim,
                args.pd_hidden_dim,
                self.history_len + self.max_delay,
            )
        else:
            raise ValueError(f"Unsupported RDC predictor_model: {args.predictor_model}")

        self.regression_head = nn.Linear(args.pd_hidden_dim, len(self.layout.continuous_indices))
        self.classification_head = nn.Linear(
            args.pd_hidden_dim, len(self.layout.binary_indices) * 3
        )

    @staticmethod
    def _gather_time(values: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        batch, _, agents = values.shape[:3]
        queries = indices.numel()
        feature_shape = values.shape[3:]
        index = indices.view(1, queries, 1, *([1] * len(feature_shape))).expand(
            batch, queries, agents, *feature_shape
        )
        return torch.gather(values, 1, index)

    def _build_history(
        self,
        delayed_obs: torch.Tensor,
        previous_actions: torch.Tensor,
        delays: torch.Tensor,
        query_indices: torch.Tensor,
    ) -> torch.Tensor:
        offsets = torch.arange(
            -self.history_len + 1,
            1,
            device=query_indices.device,
            dtype=torch.long,
        )
        history_indices = (query_indices.unsqueeze(1) + offsets.unsqueeze(0)).clamp(
            min=0, max=delayed_obs.size(1) - 1
        )
        flat_indices = history_indices.reshape(-1)
        obs_history = self._gather_time(delayed_obs, flat_indices).view(
            delayed_obs.size(0), query_indices.numel(), self.history_len,
            delayed_obs.size(2), delayed_obs.size(3)
        )
        action_history = self._gather_time(previous_actions, flat_indices).view(
            previous_actions.size(0), query_indices.numel(), self.history_len,
            previous_actions.size(2), previous_actions.size(3)
        )
        delay_history = self._gather_time(delays.unsqueeze(-1), flat_indices).view(
            delays.size(0), query_indices.numel(), self.history_len, delays.size(2), 1
        )

        delay_history = (delay_history > 0).to(delayed_obs.dtype)
        context = torch.cat((obs_history, action_history, delay_history), dim=-1)
        return context.permute(0, 1, 3, 2, 4).contiguous()

    @staticmethod
    def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        expanded = mask.expand_as(values).to(values.dtype)
        return (values * expanded).sum() / expanded.sum().clamp_min(1.0)

    def forward(
        self,
        delayed_obs: torch.Tensor,
        delays: torch.Tensor,
        actions_onehot: torch.Tensor,
        query_indices: torch.Tensor | None = None,
        clean_obs: torch.Tensor | None = None,
        valid_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        batch, time_steps, agents, obs_dim = delayed_obs.shape
        if query_indices is None:
            query_indices = torch.arange(time_steps, device=delayed_obs.device)
        else:
            query_indices = query_indices.to(delayed_obs.device, dtype=torch.long)

        previous_actions = torch.zeros_like(actions_onehot)
        previous_actions[:, 1:] = actions_onehot[:, :-1]
        prediction = self._gather_time(delayed_obs, query_indices).clone()
        query_delays = self._gather_time(delays.unsqueeze(-1), query_indices).squeeze(-1)
        query_delays = query_delays.clamp(min=0, max=self.max_delay)
        query_actions = self._gather_time(previous_actions, query_indices)
        context = self._build_history(
            delayed_obs,
            previous_actions,
            delays,
            query_indices,
        )
        if self.args.predictor_model == "gru":
            hidden = self.backbone.encode(context)
            target_sequence = None
        else:
            hidden = None
            target_sequence = context[..., :obs_dim].clone()

        if valid_mask is None:
            loss_mask = prediction.new_ones(batch, query_indices.numel(), agents, 1)
        else:
            selected_mask = self._gather_time(valid_mask.unsqueeze(2), query_indices)
            loss_mask = selected_mask.expand(-1, -1, agents, -1).to(prediction.dtype)

        regression_loss = prediction.new_zeros(())
        classification_loss = prediction.new_zeros(())
        current_times = query_indices.view(1, -1, 1).expand(batch, -1, agents)
        source_times = current_times - query_delays

        for step in range(self.max_delay):
            if self.args.predictor_model == "transformer":
                hidden = self.backbone(context, target_sequence)
            regression = self.regression_head(hidden)
            classification_logits = self.classification_head(hidden).view(
                batch, query_indices.numel(), agents, self.binary_indices.numel(), 3
            )

            active = (query_delays > step).unsqueeze(-1)
            next_prediction = prediction.clone()
            if self.continuous_indices.numel() > 0:
                current_continuous = prediction.index_select(-1, self.continuous_indices)
                updated = current_continuous + regression
                next_prediction[..., self.continuous_indices] = torch.where(
                    active, updated, current_continuous
                )
            if self.binary_indices.numel() > 0:
                current_binary = prediction.index_select(-1, self.binary_indices)
                class_delta = classification_logits.argmax(dim=-1).to(prediction.dtype) - 1.0
                updated = (current_binary + class_delta).clamp(0.0, 1.0)
                next_prediction[..., self.binary_indices] = torch.where(
                    active, updated, current_binary
                )

            if clean_obs is not None:
                target_times = torch.minimum(source_times + step + 1, current_times)
                target = self._gather_time_per_agent(clean_obs, target_times)
                if self.continuous_indices.numel() > 0:
                    target_delta = target.index_select(-1, self.continuous_indices) - prediction.index_select(
                        -1, self.continuous_indices
                    )
                    regression_loss = regression_loss + self._masked_mean(
                        (regression - target_delta).pow(2), loss_mask
                    )
                if self.binary_indices.numel() > 0:
                    target_delta = target.index_select(-1, self.binary_indices) - prediction.index_select(
                        -1, self.binary_indices
                    )
                    target_classes = target_delta.round().clamp(-1, 1).long() + 1
                    ce = F.cross_entropy(
                        classification_logits.reshape(-1, 3),
                        target_classes.reshape(-1),
                        reduction="none",
                    ).view(batch, query_indices.numel(), agents, -1)
                    classification_loss = classification_loss + self._masked_mean(ce, loss_mask)
            prediction = next_prediction

            if step + 1 < self.max_delay:
                remaining_delay = (query_delays - step - 1 > 0).unsqueeze(-1).to(
                    delayed_obs.dtype
                )
                echo_input = torch.cat(
                    (prediction, query_actions, remaining_delay), dim=-1
                )
                if self.args.predictor_model == "gru":
                    hidden = self.backbone.step(echo_input, hidden)
                else:
                    context = torch.cat((context, echo_input.unsqueeze(3)), dim=3)
                    target_sequence = torch.cat(
                        (target_sequence, prediction.unsqueeze(3)), dim=3
                    )

        divisor = max(self.max_delay, 1)
        regression_loss = regression_loss / divisor
        classification_loss = classification_loss / divisor
        if clean_obs is not None:
            clean_target = self._gather_time(clean_obs, query_indices)
            observation_loss = self._masked_mean((prediction - clean_target).pow(2), loss_mask)
        else:
            observation_loss = prediction.new_zeros(())
        return {
            "predicted_obs": prediction,
            "regression_loss": regression_loss,
            "classification_loss": classification_loss,
            "observation_loss": observation_loss,
        }

    @staticmethod
    def _gather_time_per_agent(values: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        index = indices.unsqueeze(-1).expand(*indices.shape, values.size(-1))
        return torch.gather(values, 1, index)

import torch
import torch.nn as nn

from modules.bcrbc.agent_readout import AgentReadout
from modules.bcrbc.block_causal_transformer import BlockCausalTransformer
from modules.bcrbc.dynamics_tokenizer import DynamicsTokenizer
from modules.bcrbc.observation_tokenizer import ObservationDecoder, ObservationEncoder
from modules.bcrbc.q_head import BCRBCQHead


class BCRBCModel(nn.Module):
    def __init__(self, observation_dim, n_actions, n_agents, args):
        super().__init__()

        model_hidden_dim = args.bcrbc_d_model
        agent_output_dim = args.bcrbc_agent_output_dim
        transformer_depth = args.bcrbc_depth
        attention_heads = args.bcrbc_heads
        z_dim = args.bcrbc_z_dim
        num_z_tokens = args.bcrbc_num_z_tokens

        self.z_dim = z_dim
        self.num_z_tokens = num_z_tokens
        self.observation_dim = observation_dim
        self.context_window = args.bcrbc_context_window
        self.flow_steps = args.bcrbc_flow_steps

        self.observation_encoder = ObservationEncoder(
            observation_dim,
            model_hidden_dim,
            z_dim,
            num_z_tokens,
            transformer_depth,
            attention_heads,
            context_window=self.context_window,
        )
        self.observation_decoder = ObservationDecoder(
            observation_dim,
            model_hidden_dim,
            z_dim,
            num_z_tokens,
            transformer_depth,
            attention_heads,
            context_window=self.context_window,
        )

        max_time_steps = args.env_info["episode_limit"] + 2
        self.dynamics_tokenizer = DynamicsTokenizer(
            z_dim,
            n_actions,
            n_agents,
            model_hidden_dim,
            max_t=max_time_steps,
            num_z_tokens=num_z_tokens,
        )
        self.transformer = BlockCausalTransformer(
            model_hidden_dim,
            transformer_depth,
            attention_heads,
            dropout=args.bcrbc_dropout,
            agent_slice=self.dynamics_tokenizer.query_slice,
            context_window=self.context_window,
        )
        self.z_output_norm = nn.LayerNorm(model_hidden_dim)
        self.z_predictor = nn.Linear(model_hidden_dim, z_dim)
        self.agent_readout = AgentReadout(model_hidden_dim, agent_output_dim)
        self.q_head = BCRBCQHead(agent_output_dim, n_actions, args.bcrbc_q_hidden_dim)

    def encode_observations(self, observations, **kwargs):
        return self.observation_encoder(observations, **kwargs)

    def decode_observations(self, z, **kwargs):
        return self.observation_decoder(z, **kwargs)

    @torch.no_grad()
    def prepare_condition(self, history_z, previous_actions, start_t=0, kv_cache=None):
        """Build detached MASK-history KV once for all current queries."""
        tokens = self.dynamics_tokenizer(
            history_z, previous_actions, torch.ones_like(history_z[..., :1, :1]),
            start_t=start_t,
        )
        return self.transformer.prepare_condition(
            tokens, start_t=start_t, kv_cache=kv_cache, detach=True,
        )

    def estimate_clean_z(
        self,
        noisy_z,
        previous_actions,
        signal_levels,
        start_t=0,
        kv_cache=None,
        use_kv_cache=False,
        rope_offset=None,
        condition=None,
    ):
        tokens = self.dynamics_tokenizer(
            noisy_z,
            previous_actions,
            signal_levels,
            start_t=start_t,
        )
        if condition is not None:
            transformer_outputs = self.transformer.query_condition(
                tokens, condition, start_t=start_t,
            )
            new_kv_cache = None
        elif use_kv_cache:
            transformer_outputs, new_kv_cache = self.transformer(
                tokens, kv_cache=kv_cache, use_kv_cache=True, rope_offset=rope_offset,
            )
        else:
            transformer_outputs = self.transformer(tokens)
            new_kv_cache = None

        z_outputs = transformer_outputs[..., self.dynamics_tokenizer.z_slice, :]
        normalized_z_outputs = self.z_output_norm(z_outputs)
        predicted_z = self.z_predictor(normalized_z_outputs)
        agent_outputs = self.agent_readout(
            transformer_outputs,
            self.dynamics_tokenizer.query_slice,
        )
        return {
            "predicted_z": predicted_z,
            "agent_outputs": agent_outputs,
            "q_values": self.q_head(agent_outputs),
            "kv_cache": new_kv_cache,
        }

    def complete_current(self, encoded_z, previous_actions, missing_mask,
                         noise, condition, start_t=0):
        """Generate independent current blocks against the same MASK history.

        Linear flow: x(s) = (1-s) noise + s target. The network predicts the
        clean endpoint, giving velocity (predicted_z - x) / (1-s).
        Solver iterations and Q share history; generated values never enter it.
        """
        current_z = torch.where(missing_mask, noise, encoded_z)
        if missing_mask.any():
            for index in range(self.flow_steps):
                signal = torch.full_like(
                    encoded_z[..., :1, :1], index / self.flow_steps,
                )
                signal = torch.where(missing_mask, signal, 1.0)
                estimate = self.estimate_clean_z(
                    current_z, previous_actions, signal,
                    start_t=start_t, condition=condition,
                )["predicted_z"]
                # dt / (1-s) = 1 / (K-index); last step reaches the endpoint.
                velocity_step = (estimate - current_z) / (self.flow_steps - index)
                updated_z = current_z + velocity_step
                current_z = torch.where(missing_mask, updated_z, encoded_z)
        output = self.estimate_clean_z(
            current_z, previous_actions,
            torch.ones_like(encoded_z[..., :1, :1]), start_t=start_t,
            condition=condition,
        )
        output["z"] = current_z
        return output

    def forward_training(self, observations, previous_actions,
                         missing_mask, start_t=0, completion_noise=None,
                         compute_aux=True):
        """Encode masked history once and batch all current-time queries."""
        history_z = self.encode_observations(
            observations, missing_mask=missing_mask, rope_offset=start_t
        )
        if completion_noise is None:
            completion_noise = torch.randn_like(history_z)
        condition = self.prepare_condition(history_z, previous_actions, start_t=start_t)
        output = self.complete_current(
            history_z, previous_actions, missing_mask, completion_noise,
            condition, start_t=start_t,
        )
        output.update({
            "history_z": history_z,
            "missing_mask": missing_mask,
            "completion_noise": completion_noise,
        })
        if compute_aux:
            target_z = self.encode_observations(observations, rope_offset=start_t)
            # Clean targets enter only this auxiliary query, never Q or history.
            signal = torch.rand_like(history_z[..., :1, :1])
            noisy = torch.lerp(torch.randn_like(history_z), target_z.detach(), signal)
            noisy = torch.where(missing_mask, noisy, history_z.detach())
            flow = self.estimate_clean_z(
                noisy, previous_actions, torch.where(missing_mask, signal, 1.0),
                start_t=start_t, condition=condition,
            )
            output.update({
                "target_z": target_z.detach(),
                "predicted_z": flow["predicted_z"],
                "reconstructed_observations": self.decode_observations(
                    target_z, rope_offset=start_t,
                ),
                "masked_reconstructed_observations": self.decode_observations(
                    history_z, rope_offset=start_t,
                ),
                "generated_reconstructed_observations": self.decode_observations(
                    output["z"], history_z=history_z, rope_offset=start_t,
                ),
            })
        return output

    def forward(
        self,
        observations,
        previous_actions,
        start_t=0,
        z_override=None,
        noisy_z=None,
        signal_levels=None,
        kv_cache=None,
        use_kv_cache=False,
        rope_offset=None,
        reconstruct=True,
    ):
        z = (
            self.encode_observations(observations)
            if z_override is None
            else z_override
        )
        if noisy_z is None:
            noisy_z = z
        if signal_levels is None:
            signal_levels = z.new_ones(*z.shape[:3], 1, 1)

        dynamics_output = self.estimate_clean_z(
            noisy_z,
            previous_actions,
            signal_levels,
            start_t=start_t,
            kv_cache=kv_cache,
            use_kv_cache=use_kv_cache,
            rope_offset=rope_offset,
        )
        dynamics_output["z"] = z
        if reconstruct:
            dynamics_output["reconstructed_observations"] = self.decode_observations(z)
        return dynamics_output

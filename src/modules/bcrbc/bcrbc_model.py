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
            num_messages_per_agent=n_agents - 1,
            message_dim=observation_dim,
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
        self.message_decoder = nn.Sequential(
            nn.LayerNorm(agent_output_dim),
            nn.Linear(agent_output_dim, agent_output_dim),
            nn.ReLU(inplace=True),
            nn.Linear(agent_output_dim, observation_dim),
        )

    def encode_observations(self, observations, **kwargs):
        return self.observation_encoder(observations, **kwargs)

    def decode_observations(self, z, **kwargs):
        return self.observation_decoder(z, **kwargs)

    def estimate_clean_z(
        self,
        noisy_z,
        previous_actions,
        messages,
        signal_levels,
        start_t=0,
        kv_cache=None,
        use_kv_cache=False,
        rope_offset=None,
        history_z=None,
    ):
        tokens = self.dynamics_tokenizer(
            noisy_z,
            previous_actions,
            signal_levels,
            start_t=start_t,
            messages=messages,
        )
        if history_z is not None:
            history_tokens = self.dynamics_tokenizer(
                history_z, previous_actions, torch.ones_like(signal_levels),
                start_t=start_t, messages=messages,
            )
            transformer_outputs = self.transformer.forward_conditioned(
                tokens, history_tokens, start_t=start_t
            )
            new_kv_cache = None
        elif use_kv_cache:
            transformer_outputs, new_kv_cache = self.transformer(
                tokens,
                kv_cache=kv_cache,
                use_kv_cache=True,
                rope_offset=rope_offset,
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

    def forward_training(self, observations, previous_actions, messages,
                         missing_mask, start_t=0):
        """Train decisions directly on the masked observation encoding."""
        z = self.encode_observations(observations, rope_offset=start_t)
        history_z = self.encode_observations(
            observations, missing_mask=missing_mask, rope_offset=start_t
        )
        signal_levels = torch.ones_like(history_z[..., :1, :1])
        output = self.estimate_clean_z(
            history_z, previous_actions, messages, signal_levels,
            start_t=start_t,
        )
        reconstructed_observations = self.decode_observations(z)
        masked_reconstructed_observations = self.decode_observations(history_z)
        output.update({
            "z": history_z,
            "history_z": history_z,
            "reconstructed_observations": reconstructed_observations,
            "masked_reconstructed_observations": masked_reconstructed_observations,
            "reconstructed_messages": self.message_decoder(output["agent_outputs"]),
        })
        return output

    def forward(
        self,
        observations,
        previous_actions,
        start_t=0,
        messages=None,
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
            messages,
            signal_levels,
            start_t=start_t,
            kv_cache=kv_cache,
            use_kv_cache=use_kv_cache,
            rope_offset=rope_offset,
        )
        dynamics_output.update(
            {
                "z": z,
                "reconstructed_messages": self.message_decoder(
                    dynamics_output["agent_outputs"]
                ),
            }
        )
        if reconstruct:
            dynamics_output["reconstructed_observations"] = self.decode_observations(z)
        return dynamics_output

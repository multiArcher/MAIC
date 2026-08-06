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
        )
        self.observation_decoder = ObservationDecoder(
            observation_dim,
            model_hidden_dim,
            z_dim,
            num_z_tokens,
            transformer_depth,
            attention_heads,
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

    def encode_observations(self, observations):
        return self.observation_encoder(observations)

    def decode_observations(self, z):
        return self.observation_decoder(z)

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
    ):
        tokens = self.dynamics_tokenizer(
            noisy_z,
            previous_actions,
            signal_levels,
            start_t=start_t,
            messages=messages,
        )
        if use_kv_cache:
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
        predicted_z = self.z_predictor(self.z_output_norm(z_outputs))   # Estimated pure z.
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

    @torch.no_grad()
    def sample_z(
        self,
        previous_actions,
        messages,
        steps,
        start_t=0,
        kv_cache=None,
        rope_offset=None,
    ):
        batch_size, time_steps, num_agents = previous_actions.shape[:3]
        noisy_z = torch.randn(
            batch_size,
            time_steps,
            num_agents,
            self.num_z_tokens,
            self.z_dim,
            device=previous_actions.device,
        )
        step_size = 1.0 / steps
        for step in range(steps):
            signal_level = step * step_size
            signal_levels = noisy_z.new_full(
                (batch_size, time_steps, num_agents, 1, 1),
                signal_level,
            )
            predicted_z = self.estimate_clean_z(
                noisy_z,
                previous_actions,
                messages,
                signal_levels,
                start_t=start_t,
                kv_cache=kv_cache,
                use_kv_cache=True,
                rope_offset=rope_offset,
            )["predicted_z"]
            velocity = (predicted_z - noisy_z) / (1.0 - signal_level)
            noisy_z = noisy_z + step_size * velocity
        return noisy_z

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
                "reconstructed_observations": self.decode_observations(z),
                "reconstructed_messages": self.message_decoder(
                    dynamics_output["agent_outputs"]
                ),
            }
        )
        return dynamics_output

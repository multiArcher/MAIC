import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

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

    def complete_step(self, encoded_z, previous_actions, messages, missing_mask,
                      noise, start_t=0, kv_cache=None):
        """Generate one missing time block; only the final clean pass commits KV.

        Linear flow: x(s) = (1-s) noise + s target. The network predicts the
        clean endpoint, giving velocity (predicted_z - x) / (1-s).
        All solver iterations read the same past, never each other's KV.
        """
        current_z = torch.where(missing_mask, noise, encoded_z)
        if missing_mask.any():
            for index in range(self.flow_steps):
                signal = torch.full_like(
                    encoded_z[..., :1, :1], index / self.flow_steps,
                )
                signal = torch.where(missing_mask, signal, 1.0)
                estimate = self.estimate_clean_z(
                    current_z, previous_actions, messages, signal,
                    start_t=start_t, kv_cache=kv_cache,
                    use_kv_cache=True, rope_offset=start_t,
                )["predicted_z"]
                # dt / (1-s) = 1 / (K-index); last step reaches the endpoint.
                velocity_step = (estimate - current_z) / (self.flow_steps - index)
                updated_z = current_z + velocity_step
                current_z = torch.where(missing_mask, updated_z, encoded_z)
        output = self.estimate_clean_z(
            current_z, previous_actions, messages,
            torch.ones_like(encoded_z[..., :1, :1]), start_t=start_t,
            kv_cache=kv_cache, use_kv_cache=True, rope_offset=start_t,
        )
        output["z"] = current_z
        return output

    def _detach_cache(self, cache):
        """Keep the causal window and truncate gradients between time steps."""
        if self.context_window:
            cache = [
                (key[..., -self.context_window:, :],
                 value[..., -self.context_window:, :])
                for key, value in cache
            ]
        return [(key.detach(), value.detach()) for key, value in cache]

    def _training_step(self, encoded, target, actions, messages, missing, noise,
                       dynamics_cache, decoder_cache, start_t, compute_aux):
        """Train the current block against a detached, completed history."""
        output = self.complete_step(
            encoded, actions, messages, missing, noise,
            start_t=start_t, kv_cache=dynamics_cache,
        )
        if compute_aux:
            # Teacher endpoints enter this auxiliary pass only, never Q or KV.
            signal = torch.rand_like(encoded[..., :1, :1])
            noisy = torch.lerp(torch.randn_like(encoded), target, signal)
            noisy = torch.where(missing, noisy, encoded.detach())
            flow = self.estimate_clean_z(
                noisy, actions, messages,
                torch.where(missing, signal, 1.0),
                start_t=start_t, kv_cache=dynamics_cache,
                use_kv_cache=True, rope_offset=start_t,
            )
            decoded, decoder_cache = self.decode_observations(
                output["z"], kv_cache=decoder_cache,
                use_kv_cache=True, rope_offset=start_t,
            )
            output["predicted_z"] = flow["predicted_z"]
            output["generated_reconstructed_observations"] = decoded
            output["decoder_cache"] = decoder_cache
        return output

    def forward_training(self, observations, previous_actions, messages,
                         missing_mask, start_t=0, completion_noise=None,
                         compute_aux=True):
        """Train Q on the same sequential noise-to-clean rollout used online."""
        history_z = self.encode_observations(
            observations, missing_mask=missing_mask, rope_offset=start_t
        )
        if completion_noise is None:
            completion_noise = torch.randn_like(history_z)
        target_z = None
        if compute_aux:
            target_z = self.encode_observations(observations, rope_offset=start_t)
        dynamics_cache = decoder_cache = None
        outputs, flow_predictions, reconstructed = [], [], []
        for step in range(observations.shape[1]):
            current = slice(step, step + 1)
            encoded = history_z[:, current]
            actions = previous_actions[:, current]
            step_messages = None if messages is None else messages[:, current]
            missing = missing_mask[:, current]
            target = target_z[:, current].detach() if compute_aux else None
            step_args = (encoded, target, actions, step_messages, missing,
                         completion_noise[:, current], dynamics_cache,
                         decoder_cache, start_t + step, compute_aux)
            if torch.is_grad_enabled():
                # Recompute only the current block's solver during backward.
                output = checkpoint(
                    self._training_step, *step_args, use_reentrant=False,
                )
            else:
                output = self._training_step(*step_args)
            if compute_aux:
                flow_predictions.append(output["predicted_z"])
                reconstructed.append(output["generated_reconstructed_observations"])
                decoder_cache = self._detach_cache(output.pop("decoder_cache"))
            dynamics_cache = self._detach_cache(output.pop("kv_cache"))
            outputs.append(output)
        output = {
            key: torch.cat([item[key] for item in outputs], dim=1)
            for key in ("z", "q_values", "agent_outputs")
        }
        output.update({
            "history_z": history_z,
            "missing_mask": missing_mask,
            "completion_noise": completion_noise,
        })
        if compute_aux:
            output.update({
                "target_z": target_z.detach(),
                "predicted_z": torch.cat(flow_predictions, dim=1),
                "reconstructed_observations": self.decode_observations(
                    target_z, rope_offset=start_t,
                ),
                "masked_reconstructed_observations": self.decode_observations(
                    history_z, rope_offset=start_t,
                ),
                "generated_reconstructed_observations": torch.cat(reconstructed, dim=1),
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

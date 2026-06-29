import torch
import torch.nn as nn

from modules.bcrbc.belief_readout import BeliefReadout
from modules.bcrbc.block_causal_transformer import BlockCausalTransformer
from modules.bcrbc.block_builder import BlockBuilder
from modules.bcrbc.delay_tokenizer import DelayTokenizer
from modules.bcrbc.flow_dynamics import FlowDynamics
from modules.bcrbc.q_head import BCRBCQHead


class BCRBCDynamicsCore(nn.Module):
    """No-delay BC-RBC core: joint block-causal dynamics plus local Q heads."""

    def __init__(self, obs_dim: int, n_actions: int, n_agents: int, args, dynamic_obs_dim: int):
        super().__init__()
        # All hyperparameters are read directly from the run config (fail-loud: a
        # missing key raises here instead of silently using a fallback default).
        model_hidden_dim = args.bcrbc_d_model
        belief_dim = args.bcrbc_belief_dim
        num_transformer_layers = args.bcrbc_depth
        num_attention_heads = args.bcrbc_heads
        dropout = args.bcrbc_dropout
        # env_info is set by the runner before any module is built, so episode_limit
        # is always available; +1 covers the terminal observation step.
        max_time_steps = args.env_info["episode_limit"] + 1

        # dynamic_obs_dim is the raw environment observation (no agent-id / last-action
        # augmentation): exactly the part the obs decoder must reconstruct.
        latent_dim = args.bcrbc_z_dim
        num_latent_tokens = args.bcrbc_n_latent_tokens
        self.latent_dim = latent_dim
        self.num_latent_tokens = num_latent_tokens
        self.obs_dim = obs_dim
        self.dynamic_obs_dim = dynamic_obs_dim

        # Encoder E: augmented obs input -> num_latent_tokens bottleneck latents z
        # (homogeneous capacity tokens, like DreamerV4 spatial tokens). z is what
        # feeds the obs-token slots of the world model, what the dynamics
        # predicts/generates, and what the obs decoder reconstructs.
        self.latent_encoder = nn.Sequential(
            nn.Linear(obs_dim, model_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(model_hidden_dim, latent_dim * num_latent_tokens),
        )
        # Tokenizer consumes the bottleneck latents z (dim latent_dim) in the obs slots.
        # Messages carry raw sender observations, so message_dim is the dynamic obs
        # dim; each receiver has n_agents - 1 senders.
        self.tokenizer = DelayTokenizer(
            latent_dim, n_actions, n_agents, model_hidden_dim,
            max_t=max_time_steps + 1,
            num_messages_per_agent=n_agents - 1, message_dim=dynamic_obs_dim,
            num_latent_tokens=num_latent_tokens,
        )
        self.block_builder = BlockBuilder(self.tokenizer)

        agent_slice = self.block_builder.agent_slice

        self.transformer = BlockCausalTransformer(
            model_hidden_dim, num_transformer_layers, num_attention_heads,
            dropout=dropout, agent_slice=agent_slice,
        )
        self.readout = BeliefReadout(model_hidden_dim, belief_dim)
        self.q_head = BCRBCQHead(belief_dim, n_actions, args.bcrbc_q_hidden_dim)
        # Decoder anchor: reconstruct raw obs from the bottleneck latents z
        # (keeps z information-bearing when rec_loss_weight > 0). Consumes all
        # num_latent_tokens latents of an agent jointly.
        self.obs_decoder = nn.Sequential(
            nn.LayerNorm(latent_dim * num_latent_tokens),
            nn.Linear(latent_dim * num_latent_tokens, belief_dim),
            nn.ReLU(inplace=True),
            nn.Linear(belief_dim, dynamic_obs_dim),
        )
        self.msg_decoder = nn.Sequential(
            nn.LayerNorm(belief_dim),
            nn.Linear(belief_dim, belief_dim),
            nn.ReLU(inplace=True),
            nn.Linear(belief_dim, dynamic_obs_dim),
        )
        self.latent_predictor = nn.Sequential(
            nn.LayerNorm(model_hidden_dim),
            nn.Linear(model_hidden_dim, model_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(model_hidden_dim, model_hidden_dim),
        )
        # Flow-matching world model: predicts/generates the next-step bottleneck
        # latent z_{t+1} conditioned on the per-agent belief summary at step t.
        self.flow_dynamics = FlowDynamics(
            context_dim=belief_dim,
            latent_dim=latent_dim,
            hidden_dim=args.bcrbc_flow_hidden_dim,
            flow_time_embed_dim=args.bcrbc_flow_tau_embed_dim,
            num_flow_time_buckets=args.bcrbc_flow_tau_buckets,
            num_latent_tokens=num_latent_tokens,
        )

    def forward(self, obs, last_actions, start_t=0, messages=None, z_override=None):
        # Encode augmented obs into num_latent_tokens bottleneck latents
        # z [B, T, n, num_token, latent_dim]. z_override (same shape) lets the eval
        # rollout substitute generated latents for not-yet-arrived observations.
        batch_size, time_steps, num_agents = obs.shape[:3]
        z = self.latent_encoder(obs).reshape(
            batch_size, time_steps, num_agents, self.num_latent_tokens, self.latent_dim
        )
        if z_override is not None:
            z = z_override
        tokens = self.block_builder(z, last_actions, start_t=start_t, messages=messages)
        latents = self.transformer(tokens)
        query_indices = self.block_builder.query_indices.to(obs.device)
        beliefs = self.readout(latents, query_indices)  # [b, t, n, 1, belief_dim]
        q_values = self.q_head(beliefs)  # [b, t, n, 1, n_actions] (vector axis native)
        # Decode obs from the agent's latents jointly (flatten the token axis).
        z_flat = z.reshape(batch_size, time_steps, num_agents, self.num_latent_tokens * self.latent_dim)
        return {
            "q_values": q_values,
            "beliefs": beliefs,
            "latents": latents,
            "z": z,
            "recon_obs": self.obs_decoder(z_flat).unsqueeze(-2),
            "recon_msg": self.msg_decoder(beliefs),
            "pred_latents": self.latent_predictor(latents),
        }

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

    def __init__(self, obs_dim: int, n_actions: int, n_agents: int, args, recon_obs_dim: int | None = None):
        super().__init__()
        d_model = getattr(args, "bcrbc_d_model", 128)
        belief_dim = getattr(args, "bcrbc_belief_dim", d_model)
        depth = getattr(args, "bcrbc_depth", 2)
        heads = getattr(args, "bcrbc_heads", 4)
        dropout = getattr(args, "bcrbc_dropout", 0.0)
        episode_limit = getattr(args, "episode_limit", None)
        if episode_limit is None and hasattr(args, "env_info"):
            episode_limit = args.env_info["episode_limit"]
        max_t = int(episode_limit or 256) + 1
        max_delay = getattr(args, "bcrbc_max_delay", getattr(args, "max_delay", 16))
        self.max_delay = max_delay

        recon_obs_dim = recon_obs_dim or obs_dim
        z_dim = getattr(args, "bcrbc_z_dim", 64)
        n_latent_tokens = getattr(args, "bcrbc_n_latent_tokens", 4)
        self.z_dim = z_dim
        self.n_latent_tokens = n_latent_tokens
        self.obs_dim = obs_dim
        self.recon_obs_dim = recon_obs_dim

        # Encoder E: augmented obs input -> num_latent_tokens bottleneck latents z
        # (homogeneous capacity tokens, like DreamerV4 spatial tokens). z is what
        # feeds the obs-token slots of the world model, what the dynamics
        # predicts/generates, and what the obs decoder reconstructs.
        self.latent_encoder = nn.Sequential(
            nn.Linear(obs_dim, d_model),
            nn.ReLU(inplace=True),
            nn.Linear(d_model, z_dim * n_latent_tokens),
        )
        # Tokenizer consumes the bottleneck latents z (dim z_dim) in the obs slots.
        self.tokenizer = DelayTokenizer(
            z_dim, n_actions, n_agents, d_model, max_t=max_t + 1, max_delay=max_delay,
            d_msg=recon_obs_dim, n_latent_tokens=n_latent_tokens,
        )
        self.block_builder = BlockBuilder(self.tokenizer)

        agent_slice = self.block_builder.agent_slice

        self.transformer = BlockCausalTransformer(d_model, depth, heads, dropout=dropout, agent_slice=agent_slice)
        self.readout = BeliefReadout(d_model, belief_dim)
        self.q_head = BCRBCQHead(belief_dim, n_actions, getattr(args, "bcrbc_q_hidden_dim", belief_dim))
        # Decoder anchor: reconstruct raw obs from the bottleneck latents z
        # (keeps z information-bearing when rec_loss_weight > 0). Consumes all
        # num_latent_tokens latents of an agent jointly.
        self.obs_decoder = nn.Sequential(
            nn.LayerNorm(z_dim * n_latent_tokens),
            nn.Linear(z_dim * n_latent_tokens, belief_dim),
            nn.ReLU(inplace=True),
            nn.Linear(belief_dim, recon_obs_dim),
        )
        self.msg_decoder = nn.Sequential(
            nn.LayerNorm(belief_dim),
            nn.Linear(belief_dim, belief_dim),
            nn.ReLU(inplace=True),
            nn.Linear(belief_dim, recon_obs_dim),
        )
        self.latent_predictor = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.ReLU(inplace=True),
            nn.Linear(d_model, d_model),
        )
        self.arr_head = nn.Linear(belief_dim, max_delay + 1)
        # Flow-matching world model: predicts/generates the next-step bottleneck
        # latent z_{t+1} conditioned on the per-agent belief summary at step t.
        self.flow_dynamics = FlowDynamics(
            context_dim=belief_dim,
            z_dim=z_dim,
            tau_embed_dim=getattr(args, "bcrbc_flow_tau_embed_dim", 32),
            n_tau_buckets=getattr(args, "bcrbc_flow_tau_buckets", 64),
            n_tokens=n_latent_tokens,
        )

    def forward(self, obs, last_actions, start_t=0, obs_delay=None, obs_gen_t=None,
                obs_fresh_mask=None, messages=None, msg_gen_t=None, msg_arrive_t=None,
                msg_delay=None, msg_fresh_mask=None, z_override=None):
        # Encode augmented obs into num_latent_tokens bottleneck latents
        # z [B, T, n, num_token, z_dim]. z_override (same shape) lets the eval
        # rollout substitute generated latents for not-yet-arrived observations.
        b, t, n = obs.shape[:3]
        z = self.latent_encoder(obs).reshape(b, t, n, self.n_latent_tokens, self.z_dim)
        if z_override is not None:
            z = z_override
        tokens = self.block_builder(z, last_actions, start_t=start_t, obs_delay=obs_delay,
                                    obs_gen_t=obs_gen_t, obs_fresh_mask=obs_fresh_mask,
                                    messages=messages, msg_gen_t=msg_gen_t, msg_arrive_t=msg_arrive_t,
                                    msg_delay=msg_delay, msg_fresh_mask=msg_fresh_mask)
        latents = self.transformer(tokens)
        query_indices = self.block_builder.query_indices.to(obs.device)
        beliefs = self.readout(latents, query_indices)  # [b, t, n, 1, belief_dim]
        q_values = self.q_head(beliefs)  # [b, t, n, 1, n_actions] (vector axis native)
        # Decode obs from the agent's latents jointly (flatten the token axis).
        z_flat = z.reshape(b, t, n, self.n_latent_tokens * self.z_dim)
        return {
            "q_values": q_values,
            "beliefs": beliefs,
            "latents": latents,
            "z": z,
            "recon_obs": self.obs_decoder(z_flat).unsqueeze(-2),
            "recon_msg": self.msg_decoder(beliefs),
            "pred_latents": self.latent_predictor(latents),
            "delay_logits": self.arr_head(beliefs),
        }

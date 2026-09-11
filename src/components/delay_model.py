"""Unified arrival-time delay model.

Generalizes the arrival logic of :class:`components.communication_model.CommunicationModel`
to an arbitrary payload. BCRBC uses this component for local observation delay.

Arrival semantics (the fix vs. the old query-centric paths): each item produced at
``sent_time = t`` is assigned a delay sampled **once** for that packet, giving a fixed
``arrival = t + delay``. A query at step ``q`` receives, per source, the payload of the
**freshest** item with ``arrival <= q``. If nothing has arrived yet the slot is
zero-filled and flagged unarrived (``gen_t = -1``, ``fresh = 0``). Because availability
is decided by arrival time (not a fresh per-query draw), delivered information is
monotone in ``q`` — a received item never "un-arrives".

Locked design: training is always no-delay (``training=True`` ⇒ delay 0); delay is
never exposed to the model as an input (staleness lives in the delivered content);
the wrapper supplies zero-filled missing packets and the BCRBC encoder replaces
missing observation slots with its learned MASK token.

Layout convention: ``payload`` is ``[b, T, *source_dims, *feat_dims]`` where the
leading axis ``b`` is the batch/env axis (single env ``b=1``, parallel ``b=N``, replay
``b=B`` — identical code), ``T`` is the (sent==query) time axis, the middle
``source_dims`` carry an independent delay each, and the trailing ``feat_dims`` (count
given by ``feat_ndims``) are copied verbatim.
"""

import math

import torch
import torch.nn as nn


class DelayModel(nn.Module):
    def __init__(
        self,
        delay_type: str = "gaussian",
        delay_mean: float = 1.0,
        delay_std: float = 1.0,
        max_delay: int = 16,
        delay_per_source: bool = True,
    ):
        super().__init__()
        if delay_type not in {"uniform", "gaussian"}:
            # "fixed" is intentionally dropped: a fixed delay d is N(d, 0).
            raise ValueError(f"Unknown delay_type: {delay_type}")
        self.delay_type = delay_type
        self.delay_mean = float(delay_mean)
        self.delay_std = float(delay_std)
        self.max_delay = int(max_delay)
        self.delay_per_source = bool(delay_per_source)
        self.generator = None

        # Stateful (online) caches; allocated by reset(). Shapes are established on the
        # first push so the model stays payload-agnostic.
        self._cache_payload = None   # [b, max_t, *src, *feat]
        self._cache_arrival = None   # [b, max_t, *src]  (float; +inf == never arrives)
        self._cache_sent = None      # [b, max_t, *src]  (long; -1 == empty slot)
        self._max_t = 0
        self._feat_ndims = 1

    # ------------------------------------------------------------------ sampling
    def _sample_delay(self, shape, device, training: bool) -> torch.Tensor:
        """Sample an integer delay per (sent step, source). training ⇒ 0."""
        if training:
            return torch.zeros(shape, device=device, dtype=torch.long)

        return self._draw(shape, device)

    def _draw(self, shape, device) -> torch.Tensor:
        if self.delay_type == "uniform":
            sampled = torch.randint(0, self.max_delay + 1, shape, device=device, generator=self.generator)
        else:  # gaussian
            sampled = torch.normal(self.delay_mean, self.delay_std, size=shape, device=device, generator=self.generator).clamp(min=0.0).ceil()
        return sampled.clamp(0, self.max_delay).long()

    # ------------------------------------------------------- freshest-arrived gather
    @staticmethod
    def _query_freshest_info(
        payload: torch.Tensor,      # [b, Ts, *src, *feat]
        arrival: torch.Tensor,      # [b, Ts, *src]  float
        sent: torch.Tensor,         # [b, Ts, *src]  long, -1 == empty
        query_steps: torch.Tensor,  # [Q] absolute query times (long)
        feat_ndims: int,
    ):
        """Deliver, per (query step, source), the freshest item with arrival <= query.

        Returns (gathered [b, Q, *src, *feat], gen_t [b,Q,*src], delay [b,Q,*src],
        fresh [b,Q,*src]); unarrived slots are zero-filled with gen_t = -1.
        """
        batch_size, time_size = payload.shape[0], payload.shape[1]
        src_shape = payload.shape[2: payload.ndim - feat_ndims]
        feat_shape = payload.shape[payload.ndim - feat_ndims:]
        n_src = len(src_shape)
        Q = query_steps.shape[0]

        src_ones = (1,) * n_src
        arrival_expanded = arrival.unsqueeze(1)                # [b, 1, t, *src]
        sent_expanded = sent.unsqueeze(1)                  # [b, 1, t, *src]
        query_expanded = query_steps.view(1, Q, 1, *src_ones)  # [1, Q, 1,  *1]

        has_arrived = (arrival_expanded <= query_expanded) & (sent_expanded >= 0)  # [b, Q, t, *src]
        # Freshest == largest sent_time among arrived; -1 where none.
        valid_sent = torch.where(has_arrived, sent_expanded, -1)
        
        chosen_sent, _ = valid_sent.max(dim=2)  # [b,Q,*src], chosen timestep information was sent.
        no_msg = chosen_sent < 0

        # Map the chosen absolute sent time back to its index along Ts. sent is the
        # absolute time written at each Ts slot, so build a lookup by searching the
        # equal sent slot. Since sent[:, k] is a contiguous block of equal values per
        # source only in the obs case, do it robustly via argmax over equality.
        sent_abs = sent                                                # [b,Ts,*src]
        # match[b,Q,Ts,*src] = (sent_abs == chosen_sent)
        match = sent_abs.unsqueeze(1) == chosen_sent.unsqueeze(2)
        match = match & (sent_abs.unsqueeze(1) >= 0)
        ts_index = match.float().argmax(dim=2)  # [b,Q,*src] (0 where no_msg, index in payload matrix.)

        # Gather payload along Ts using ts_index, broadcast over feat dims.
        idx = ts_index.view(batch_size, Q, *src_shape, *([1] * feat_ndims)).expand(batch_size, Q, *src_shape, *feat_shape)
        payload_q = payload.unsqueeze(1).expand(batch_size, Q, time_size, *src_shape, *feat_shape)
        gathered = torch.gather(payload_q, 2, idx.unsqueeze(2)).squeeze(2)

        no_msg_feat = no_msg.view(batch_size, Q, *src_shape, *([1] * feat_ndims))
        gathered = gathered.masked_fill(no_msg_feat, 0.0)   # Mask when no message, max return 0 index.

        gen_t = torch.where(no_msg, -1, chosen_sent)
        q_full = query_steps.view(1, Q, *src_ones).expand(batch_size, Q, *src_shape)
        delay = torch.where(no_msg, -1, q_full - chosen_sent)
        fresh = (~no_msg) & (delay == 0)
        return gathered, gen_t, delay, fresh

    # --------------------------------------------------------------- batched window
    def forward(self, payload: torch.Tensor, start_t: int = 0, training: bool = True, feat_ndims: int = 1):
        """Stateless batched delay over a full window (used by the comm pathway).

        payload [b, T, *src, *feat] sent at absolute steps start_t .. start_t+T-1 and
        queried at the same steps. Returns (delayed_payload [b,T,*src,*feat], gen_t,
        delay, fresh), each metadata [b,T,*src].
        """
        b, T = payload.shape[0], payload.shape[1]
        device = payload.device
        src_meta_shape = payload.shape[2: payload.ndim - feat_ndims]
        sent_steps = torch.arange(start_t, start_t + T, device=device, dtype=torch.long)

        delay = self._sample_delay((b, T, *src_meta_shape), device, training)
        sent = sent_steps.view(1, T, *([1] * len(src_meta_shape))).expand(b, T, *src_meta_shape).clone()
        arrival = (sent + delay).to(torch.float32)
        return self._query_freshest_info(payload, arrival, sent, sent_steps, feat_ndims)

    # ----------------------------------------------------------------- online (obs)
    def reset(self):
        """Drop the online cache (re-allocated on the next push)."""
        self._cache_payload = None
        self._cache_arrival = None
        self._cache_sent = None
        self._max_t = 0

    def _ensure_cache(self, batch_size, src_shape, feat_shape, max_t, device, dtype, feat_ndims):
        if (self._cache_payload is None 
            or self._cache_payload.shape[0] < batch_size 
            or self._cache_payload.shape[1] < max_t
        ):
            self._feat_ndims = feat_ndims
            self._max_t = max_t
            self._cache_payload = torch.zeros(batch_size, max_t, *src_shape, *feat_shape, device=device, dtype=dtype)
            self._cache_arrival = torch.full((batch_size, max_t, *src_shape), math.inf, device=device, dtype=torch.float32)
            self._cache_sent = torch.full((batch_size, max_t, *src_shape), -1, device=device, dtype=torch.long)

    def push_step(self, payload_t: torch.Tensor, t: int, training: bool, max_t: int, feat_ndims: int = 1):
        """Store one step's payload [b_active, *src, *feat] produced at absolute step t.

        ``bs`` selects the absolute batch rows that are active this step (ragged
        rollouts); inactive rows keep their cached state.
        """
        device = payload_t.device
        src_shape = payload_t.shape[1: payload_t.ndim - feat_ndims]  # Shape of axes other than batch and data
        feat_shape = payload_t.shape[payload_t.ndim - feat_ndims:]
        batch_size = payload_t.shape[0]
        
        self._ensure_cache(batch_size, src_shape, feat_shape, max_t, device, payload_t.dtype, feat_ndims)
        
        delay = self._sample_delay((batch_size, *src_shape), device, training)
        arrival = (t + delay).to(torch.float32)
        self._cache_payload[:, t] = payload_t
        self._cache_arrival[:, t] = arrival
        self._cache_sent[:, t] = t

    @staticmethod
    def _infer_b(bs, n_active):
        if isinstance(bs, (list, tuple, torch.Tensor)):
            return max(int(i) for i in bs) + 1
        return n_active

    def query_step(self, t: int, bs=slice(None)):
        """Freshest arrived at absolute step t for the selected rows.

        Returns (payload [b_sel, *src, *feat], gen_t, delay, fresh), metadata
        [b_sel, *src]; the query-time axis is squeezed out.
        """
        prefix = slice(0, t + 1)
        payload = self._cache_payload[bs, prefix]
        arrival = self._cache_arrival[bs, prefix]
        sent = self._cache_sent[bs, prefix]
        query_steps = torch.tensor([t], device=payload.device, dtype=torch.long)
        gathered, gen_t, delay, fresh = self._query_freshest_info(payload, arrival, sent, query_steps, self._feat_ndims)
        return gathered[:, 0], gen_t[:, 0], delay[:, 0], fresh[:, 0]

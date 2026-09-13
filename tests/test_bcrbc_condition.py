import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from modules.bcrbc.block_causal_transformer import BlockCausalTransformer


def old_forward_conditioned(transformer, queries, history, start_t):
    """Original joint traversal, retained as the numerical reference."""
    steps = queries.shape[-4]
    positions = torch.arange(steps)
    distance = positions[:, None] - positions[None, :]
    causal, past = distance >= 0, distance > 0
    if transformer.context_window is not None:
        causal &= distance < transformer.context_window
        past &= distance < transformer.context_window
    query_mask = torch.cat([past, distance == 0], dim=-1)
    space_mask = transformer._build_space_mask(queries.shape[-2])
    frequencies = transformer.rotary(steps, offset=start_t)
    for layer, is_time in zip(transformer.layers, transformer.is_time_layer):
        if is_time:
            history, history_cache = layer(
                history.movedim(-4, -2), mask=causal,
                rotary_pos_emb=frequencies, return_cache=True,
            )
            queries = layer(
                queries.movedim(-4, -2), mask=query_mask,
                rotary_pos_emb=frequencies, kv_cache=history_cache,
            ).movedim(-2, -4)
            history = history.movedim(-2, -4)
        else:
            queries = layer(queries, mask=space_mask)
            history = layer(history, mask=space_mask)
    return transformer.final_norm(queries)


class ConditionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def make_transformer(self, window):
        torch.manual_seed(17)
        return BlockCausalTransformer(
            model_hidden_dim=16, num_transformer_layers=4,
            num_attention_heads=2, dropout=0.0, agent_slice=slice(2, 3),
            context_window=window, time_block_every=2,
        )

    def test_split_matches_original_conditioned_forward(self):
        for window in (None, 3):
            with self.subTest(window=window):
                model = self.make_transformer(window)
                history = torch.randn(2, 6, 2, 3, 16)
                queries = torch.randn_like(history)
                expected = old_forward_conditioned(
                    model.transformer, queries, history, start_t=5,
                )
                condition = model.prepare_condition(history, start_t=5)
                actual = model.query_condition(queries, condition, start_t=5)
                torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
                torch.testing.assert_close(
                    model.forward_conditioned(queries, history, start_t=5), expected,
                    atol=1e-6, rtol=1e-5,
                )

    def test_batch_queries_match_independent_queries(self):
        for window in (None, 3):
            with self.subTest(window=window):
                model = self.make_transformer(window)
                history = torch.randn(2, 7, 2, 3, 16)
                queries = torch.randn(2, 4, 2, 3, 16)
                condition = model.prepare_condition(history, start_t=7)
                batched = model.query_condition(queries, condition, start_t=9)
                independent = torch.cat([
                    model.query_condition(
                        queries[:, t:t + 1], condition, start_t=9 + t,
                    )
                    for t in range(queries.shape[1])
                ], dim=1)
                torch.testing.assert_close(batched, independent, atol=1e-6, rtol=1e-5)

    def test_cached_history_preparation_matches_dense(self):
        for window in (None, 3):
            with self.subTest(window=window):
                model = self.make_transformer(window)
                history = torch.randn(2, 7, 2, 3, 16)
                queries = torch.randn(2, 3, 2, 3, 16)
                dense = model.prepare_condition(history, start_t=5)
                prefix = model.prepare_condition(history[:, :4], start_t=5)
                cached = model.prepare_condition(
                    history[:, 4:], start_t=9, kv_cache=prefix["kv"],
                )
                torch.testing.assert_close(cached["positions"], torch.arange(5, 12))
                for dense_kv, cached_kv in zip(dense["kv"], cached["kv"]):
                    for expected, actual in zip(dense_kv, cached_kv):
                        torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
                torch.testing.assert_close(
                    model.query_condition(queries, cached, start_t=10),
                    model.query_condition(queries, dense, start_t=10),
                    atol=1e-6, rtol=1e-5,
                )

    def test_history_gradients_are_preserved_unless_detached(self):
        model = self.make_transformer(3)
        history = torch.randn(1, 4, 2, 3, 16, requires_grad=True)
        queries = torch.randn(1, 2, 2, 3, 16)
        condition = model.prepare_condition(history, start_t=5)
        output = model.query_condition(queries, condition, start_t=7)
        output.sum().backward()
        self.assertGreater(history.grad.abs().sum().item(), 0)
        detached = model.prepare_condition(history, start_t=5, detach=True)
        self.assertTrue(all(not value.requires_grad for kv in detached["kv"] for value in kv))


if __name__ == "__main__":
    unittest.main()

"""Smoke test for axial space-time transformer shape and output.

Tests that the ported BlockCausalTransformer:
  - accepts [B, T, N, A, D] shaped input
  - produces [B, T, N, A, D] shaped output
  - contains no NaN values
  - runs without errors

N is the agent axis and A is the per-agent local-token axis.
"""

import torch
import sys
from pathlib import Path

# Add src to path
repo_root = Path(__file__).parents[3]
sys.path.insert(0, str(repo_root / "src"))

from modules.bcrbc.block_causal_transformer import BlockCausalTransformer


def test_bcrbc_axial_shape():
    """Test basic shape and output properties."""
    batch_size = 2
    time_steps = 5
    n_agents = 3
    local_tokens = 3  # obs, action, query
    d_model = 128

    print(f"Creating BlockCausalTransformer(model_hidden_dim={d_model}, num_transformer_layers=4, num_attention_heads=4)")
    transformer = BlockCausalTransformer(
        model_hidden_dim=d_model,
        num_transformer_layers=4,
        num_attention_heads=4,
        dropout=0.0,
        agent_slice=None,
    )
    print(f"  Model created successfully")

    # Create random input
    print(f"Creating random input [{batch_size}, {time_steps}, {n_agents}, {local_tokens}, {d_model}]")
    tokens = torch.randn(batch_size, time_steps, n_agents, local_tokens, d_model)
    print(f"  Input shape: {tokens.shape}")

    # Forward pass
    print(f"Running forward pass...")
    with torch.no_grad():
        output = transformer(tokens)
    print(f"  Output shape: {output.shape}")

    # Verify output shape
    assert output.shape == tokens.shape, \
        f"Shape mismatch: expected {tokens.shape}, got {output.shape}"
    print(f"  OK: Output shape matches input shape")

    # Verify no NaN
    assert not torch.isnan(output).any(), \
        f"Output contains NaN values"
    print(f"  OK: No NaN in output")

    # Verify output dtype
    assert output.dtype == torch.float32, \
        f"Expected float32, got {output.dtype}"
    print(f"  OK: Output dtype is float32")

    print("\n✓ All checks passed!")
    return True


if __name__ == "__main__":
    try:
        test_bcrbc_axial_shape()
        print("\nTest complete: SUCCESS")
        sys.exit(0)
    except Exception as e:
        print(f"\nTest complete: FAILED")
        print(f"Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)

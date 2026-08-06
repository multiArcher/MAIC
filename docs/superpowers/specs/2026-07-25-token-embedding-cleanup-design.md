# Token Embedding Cleanup Design

## Goal

Make `DelayTokenizer.forward` use standard embedding terminology and one
uniform token-type path without changing tensor shapes or numerical behavior.

## Design

- Preserve token layout `[B, T, N, A, D]`.
- Preserve local token order `[OBS..., ACTION, MSG..., QUERY]`.
- Define the complete static token-type ID sequence once in `__init__` and
  register it as a non-persistent buffer.
- In `forward`, obtain:
  - token-type embeddings shaped `[A, D]`;
  - agent embeddings shaped `[N, 1, D]`;
  - time embeddings shaped `[1, T, 1, 1, D]`.
- Add these embeddings directly to the projected token values using
  broadcasting, then apply the existing layer normalization.
- Remove the terms `identity`, `content_identity`, and `query_identity`.

## Compatibility

The refactor must preserve:

- tokenizer parameters and output shape;
- query-token placement;
- query, agent, modality, and time contributions;
- all downstream transformer and readout interfaces.

## Verification

Run the existing BCRBC token-schema smoke test. It checks tokenizer shape,
message handling, query placement, transformer compatibility, and agent
isolation.

# SPDX-License-Identifier: Apache-2.0
# Standard
from types import SimpleNamespace

# Third Party
import msgspec

# First Party
from lmcache.v1.multiprocess.group_view import (
    EngineGroupInfo,
    expand_engine_block_ids,
    get_engine_group_indices,
    num_engine_group_infos,
    num_engine_groups,
    slice_block_ids_per_group,
)


def test_engine_group_infos_default_to_one_engine_group():
    assert num_engine_groups([]) == 1
    assert num_engine_group_infos([]) == 1
    assert get_engine_group_indices([], 1) is None


def test_engine_group_infos_build_per_layer_engine_group_indices():
    groups = [
        EngineGroupInfo(0, (0, 2)),
        EngineGroupInfo(1, (1, 3)),
    ]

    assert num_engine_groups(groups) == 2
    assert num_engine_group_infos(groups) == 2
    assert get_engine_group_indices(groups, 4) == [0, 1, 0, 1]


def test_engine_group_infos_expand_engine_block_ids():
    groups = [
        EngineGroupInfo(0, (0, 2)),
        EngineGroupInfo(0, (4,)),
        EngineGroupInfo(1, (1, 3)),
    ]

    assert expand_engine_block_ids(groups, [[10, 11], [20, 21]]) == [
        [10, 11],
        [10, 11],
        [20, 21],
    ]


def test_engine_group_info_old_payload_defaults_sw_size():
    """A pre-sw_size_tokens msgspec payload decodes with the -1 default."""
    old_payload = {"engine_group_id": 0, "layer_indices": (0, 1)}

    decoded = msgspec.msgpack.decode(
        msgspec.msgpack.encode(old_payload), type=EngineGroupInfo
    )

    assert decoded.sw_size_tokens == -1


def test_engine_group_infos_msgspec_round_trip():
    """The groups encode/decode losslessly via msgspec (the IPC path)."""
    groups = [
        EngineGroupInfo(0, (0, 2)),
        EngineGroupInfo(1, (1, 3), sw_size_tokens=128),
    ]

    decoded = msgspec.msgpack.decode(
        msgspec.msgpack.encode(groups), type=list[EngineGroupInfo]
    )

    assert decoded == groups


def test_engine_group_infos_exclude_uncovered_layers():
    """Layers not referenced by any group are tagged EXCLUDED_ENGINE_GROUP.

    Cross-layer KV-sharing layers (e.g. google/gemma-4-E4B-it) alias a target
    owner's KV cache and are intentionally left out of every group; downstream
    grouping skips them rather than treating partial coverage as an error.
    """
    # First Party
    from lmcache.v1.kv_layer_groups import EXCLUDED_ENGINE_GROUP

    groups = [
        EngineGroupInfo(0, (0,)),
        EngineGroupInfo(1, (1,)),
    ]

    # Layer 2 is not covered by any group -> excluded, not an error.
    assert get_engine_group_indices(groups, 3) == [0, 1, EXCLUDED_ENGINE_GROUP]


def test_engine_group_infos_reject_out_of_range_layer():
    groups = [EngineGroupInfo(0, (0, 5))]

    try:
        get_engine_group_indices(groups, 3)
    except ValueError as exc:
        assert "outside registered layer range" in str(exc)
    else:
        raise AssertionError("Expected out-of-range layer index to fail")


def test_slice_block_ids_uniform_block_sizes():
    """Groups sharing one tokens_per_block slice to equal counts."""
    allocated = {0: list(range(16)), 1: list(range(100, 116))}
    sliced = slice_block_ids_per_group(
        allocated,
        group_tokens_per_block=[16, 16],
        start_token_idx=0,
        end_token_idx=256,
    )
    assert sliced == [list(range(16)), list(range(100, 116))]


def test_slice_block_ids_heterogeneous_block_sizes():
    """A tokens_per_block-32 group gets half the IDs of a 16 group.

    The range [0, 256) spans 256 tokens: the tokens_per_block-16 group
    needs 16 block IDs, the tokens_per_block-32 group 8, for the same
    token span.
    """
    allocated = {0: list(range(16)), 1: list(range(8))}
    sliced = slice_block_ids_per_group(
        allocated,
        group_tokens_per_block=[16, 32],
        start_token_idx=0,
        end_token_idx=256,
    )
    assert sliced == [list(range(16)), list(range(8))]


def test_slice_block_ids_smaller_than_base_block_sizes():
    """Groups with tiny paged chunks (e.g. DeepSeek V4 compressor state,
    tokens_per_block 4/8) get proportionally more IDs over one token span."""
    allocated = {0: [0], 1: list(range(64)), 2: list(range(32))}
    sliced = slice_block_ids_per_group(
        allocated,
        group_tokens_per_block=[256, 4, 8],
        start_token_idx=0,
        end_token_idx=256,
    )
    assert sliced == [[0], list(range(64)), list(range(32))]


def test_slice_block_ids_nonzero_start_offset():
    """Start/end token offsets are divided per group by tokens_per_block."""
    allocated = {0: list(range(32)), 1: list(range(16))}
    sliced = slice_block_ids_per_group(
        allocated,
        group_tokens_per_block=[16, 32],
        start_token_idx=256,
        end_token_idx=512,
    )
    assert sliced == [list(range(16, 32)), list(range(8, 16))]


def test_slice_block_ids_missing_group_yields_empty():
    """A group with no allocated block IDs slices to an empty list."""
    allocated = {0: list(range(16))}  # group 1 absent
    sliced = slice_block_ids_per_group(
        allocated,
        group_tokens_per_block=[16, 16],
        start_token_idx=0,
        end_token_idx=256,
    )
    assert sliced == [list(range(16)), []]


def test_slice_block_ids_misaligned_range_raises():
    """A range that is not a whole number of a group's chunks is rejected."""
    allocated = {0: list(range(8)), 1: list(range(8))}
    # group 1 tokens_per_block 48; end=128 is not a multiple of 48.
    try:
        slice_block_ids_per_group(
            allocated,
            group_tokens_per_block=[16, 48],
            start_token_idx=0,
            end_token_idx=128,
        )
    except ValueError as exc:
        assert "does not align" in str(exc)
    else:
        raise AssertionError("Expected misaligned range to fail")


####
# Recurrent-state slicing: state pages gate presence, never the token
# prefix, and bypass range slicing. Regression scenario: live GDN hybrid
# (chunk 1600; attention groups 1600 tokens/block; one short-conv ring
# group at 8 tokens/block holding a single rolling state page).
####

_HYBRID_TPB = [1600, 8, 1600]
_HYBRID_RECURRENT = [False, True, False]


def test_slice_recurrent_rolling_page_null_padded():
    allocated = {0: [10, 11, 12], 1: [42], 2: [20, 21, 22]}
    sliced = slice_block_ids_per_group(
        allocated,
        group_tokens_per_block=_HYBRID_TPB,
        start_token_idx=0,
        end_token_idx=4800,
        recurrent_groups=_HYBRID_RECURRENT,
    )
    assert sliced[0] == [10, 11, 12]
    assert sliced[2] == [20, 21, 22]
    # Ring group: one slot per 8 tokens over the range; all null except the
    # final slot, carrying the request's most recent state block.
    assert sliced[1] == [0] * 599 + [42]


def test_slice_recurrent_full_table_trusted_verbatim():
    # Engine provided a full per-block table (align-mode snapshot style):
    # slice like attention, nulls included, no synthesis.
    allocated = {0: [10], 2: [20]}
    allocated[1] = [0] * 200 + [77] + [0] * 399
    sliced = slice_block_ids_per_group(
        allocated,
        group_tokens_per_block=_HYBRID_TPB,
        start_token_idx=0,
        end_token_idx=4800,
        recurrent_groups=_HYBRID_RECURRENT,
    )
    assert sliced[1] == allocated[1]


def test_slice_recurrent_group_skips_alignment_check():
    # Range aligned to the attention groups; the 8-token recurrent group
    # must not raise regardless (state snapshots are range-agnostic).
    allocated = {0: [10, 11], 1: [5], 2: [20, 21]}
    sliced = slice_block_ids_per_group(
        allocated,
        group_tokens_per_block=_HYBRID_TPB,
        start_token_idx=0,
        end_token_idx=3200,
        recurrent_groups=_HYBRID_RECURRENT,
    )
    assert sliced[1] == [0] * 399 + [5]


def test_slice_attention_misalignment_still_raises_with_recurrent_present():
    try:
        slice_block_ids_per_group(
            {0: [1], 1: [5], 2: [2]},
            group_tokens_per_block=_HYBRID_TPB,
            start_token_idx=4,
            end_token_idx=1600,
            recurrent_groups=_HYBRID_RECURRENT,
        )
    except ValueError as exc:
        assert "does not align" in str(exc)
    else:
        raise AssertionError("misaligned attention range must raise")


def test_slice_without_recurrent_flags_is_legacy_behavior():
    allocated = {0: [10, 11, 12], 1: [42], 2: [20, 21, 22]}
    sliced = slice_block_ids_per_group(
        allocated,
        group_tokens_per_block=_HYBRID_TPB,
        start_token_idx=0,
        end_token_idx=4800,
    )
    # Group 1 sliced by range (legacy): 4800/8 = 600 slots from a 1-id list.
    assert sliced == [[10, 11, 12], [42], [20, 21, 22]]


####
# Group state-page flags (public scheduler-side API): recurrent groups cover
# constant-size state pages of any cache mode, and a caller-geometry
# disagreement falls back to all-attention gating.
####


def _fake_group(spec, names):
    return SimpleNamespace(kv_cache_spec=spec, layer_names=names)


def test_recurrent_flags_cover_state_pages_including_rolling_modes():
    # First Party
    from lmcache.integration.vllm.kv_cache_groups import get_group_recurrent_flags

    class MambaSpec:  # detected by class name (documented contract)
        block_size = 1600
        mamba_cache_mode = "none"  # rolling: still a constant-size state page

    class CircularBufferSpec:
        block_size = 8

    cfg = SimpleNamespace(
        kv_cache_groups=[
            _fake_group(SimpleNamespace(block_size=1600), ["full0"]),
            _fake_group(MambaSpec(), ["gdn0"]),
            _fake_group(CircularBufferSpec(), ["ring0"]),
        ]
    )
    assert get_group_recurrent_flags(cfg) == [False, True, True]
    # Caller-provided geometry: matching count accepted, mismatch falls back.
    assert get_group_recurrent_flags(cfg, 3) == [False, True, True]
    assert get_group_recurrent_flags(cfg, 6) == []

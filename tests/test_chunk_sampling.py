"""验证文本完整性筛选与候选块抽样。"""

import pytest

from eval.chunk_sampling import looks_self_contained, sample_clean_chunks

CLEAN_CHUNK = (
    "The document index stores searchable passages alongside their source metadata. "
    "When a user asks a question, the retriever selects relevant passages and sends "
    "them to the assistant, which produces an answer with numbered citations so "
    "the user can inspect the original supporting documents."
)


def test_flags_chunk_starting_mid_sentence_as_not_self_contained():
    chunk = "and the remaining document passages are indexed by—"

    assert looks_self_contained(chunk) is False


def test_flags_chunk_ending_mid_word_as_not_self_contained():
    chunk = "Documents are split into overlapping chunks and stored with"

    assert looks_self_contained(chunk) is False


def test_flags_clean_chunk_as_self_contained():
    assert looks_self_contained(CLEAN_CHUNK) is True


def test_flags_short_greeting_as_not_self_contained():
    assert looks_self_contained("What's up?") is False


def test_sample_clean_chunks_excludes_fragments():
    chunks = [
        CLEAN_CHUNK,
        "and the remaining document passages are indexed by—",
    ]

    sampled = sample_clean_chunks(chunks, n=1, seed=0)

    assert sampled == [CLEAN_CHUNK]


def test_sample_clean_chunks_raises_when_fewer_than_n_qualify():
    chunks = [CLEAN_CHUNK]

    with pytest.raises(ValueError):
        sample_clean_chunks(chunks, n=2, seed=0)

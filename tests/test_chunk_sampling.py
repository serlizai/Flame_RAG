import pytest

from eval.chunk_sampling import looks_self_contained, sample_clean_chunks

CLEAN_CHUNK = (
    "Parliament may by law admit into the Union any new State, and may by law "
    "establish new States, provided that any such law shall not take effect "
    "unless it has been passed by both Houses of Parliament and duly assented "
    "to by the President in accordance with the provisions of this Constitution."
)


def test_flags_chunk_starting_mid_sentence_as_not_self_contained():
    chunk = "boundaries or names of existing States.—Parliament may by law—"

    assert looks_self_contained(chunk) is False


def test_flags_chunk_ending_mid_word_as_not_self_contained():
    chunk = "State or by uniting two or more States or parts of States or by"

    assert looks_self_contained(chunk) is False


def test_flags_clean_chunk_as_self_contained():
    assert looks_self_contained(CLEAN_CHUNK) is True


def test_flags_short_greeting_as_not_self_contained():
    assert looks_self_contained("What's up?") is False


def test_sample_clean_chunks_excludes_fragments():
    chunks = [
        CLEAN_CHUNK,
        "boundaries or names of existing States.—Parliament may by law—",
    ]

    sampled = sample_clean_chunks(chunks, n=1, seed=0)

    assert sampled == [CLEAN_CHUNK]


def test_sample_clean_chunks_raises_when_fewer_than_n_qualify():
    chunks = [CLEAN_CHUNK]

    with pytest.raises(ValueError):
        sample_clean_chunks(chunks, n=2, seed=0)

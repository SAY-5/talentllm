"""Phrasing robustness: rewordings of one question surface the same records."""

from talentllm import Assistant

# Four ways to ask the same thing about Noah Kim's certification.
PHRASINGS = [
    "what certification does Noah Kim hold",
    "which credential has Noah Kim earned",
    "tell me about Noah Kim's certification",
    "is Noah Kim a certified Kubernetes administrator",
]

EXPECTED_ID = "cert-301"


def test_all_phrasings_retrieve_the_supporting_record():
    assistant = Assistant()
    for phrasing in PHRASINGS:
        hits = assistant._relevant(phrasing)
        ids = {hit.record.id for hit in hits}
        assert EXPECTED_ID in ids, phrasing


def test_phrasings_agree_on_the_top_record():
    assistant = Assistant()
    tops = set()
    for phrasing in PHRASINGS:
        hits = assistant._relevant(phrasing)
        assert hits, phrasing
        tops.add(hits[0].record.id)
    # Every phrasing leads with the same supporting record.
    assert tops == {EXPECTED_ID}

"""Reader tests against the real per-read.bed format and its edge cases."""

from __future__ import annotations

import tempfile
from pathlib import Path

from rltf.io import _column_index, load_fragments

HEADER_V1 = ("#chr\tstart\tend\tread_id\tmapq\torientation\tinsert_size\tread_length\t"
             "flag\tnum_cpg\tnum_mod\tmod_cps\tunmod_cpgs\tsnp_cpgs\n")


def _rows():
    return [
        # mate pair, identical offsets -> ONE fragment (no double count)
        "chr1\t10366\t10523\tRID1\t40\t+\t157\t149\t99\t6\t5\t94,96,109,114,122\t118\t\n",
        "chr1\t10366\t10523\tRID1\t40\t-\t157\t149\t147\t6\t5\t94,96,109,114,122\t118\t\n",
        # SNP masking: snp 13,18,22 dropped -> 9 usable
        "chr1\t10470\t10608\tRID2\t39\t-\t569\t151\t147\t12\t8\t26,54,71,92,100,106,108,118\t0\t13,18,22\n",
        # low MAPQ -> dropped
        "chr1\t10324\t10471\tRID3\t3\t-\t213\t144\t83\t1\t0\t\t136\t\n",
        # conflicting mates at offset 50 -> that CpG dropped; offset 60 agrees -> kept
        "chr1\t20000\t20150\tRID4\t40\t+\t150\t150\t99\t2\t2\t50,60\t\t\n",
        "chr1\t20000\t20150\tRID4\t40\t-\t150\t150\t147\t2\t1\t60\t50\t\n",
    ]


def test_reader_semantics():
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "a.bed"
        p.write_text(HEADER_V1 + "".join(_rows()))
        frags = list(load_fragments(p, min_mapq=30))

    assert len(frags) == 3  # RID1, RID2, RID4 (RID3 mapq-filtered)
    rid1 = [f for f in frags if int(f.cpg_pos[0]) == 10460][0]
    assert rid1.n_cpg == 6  # mate-deduped, not 12
    assert set(rid1.cpg_pos[rid1.states == 1].tolist()) == {10460, 10462, 10475, 10480, 10488}
    assert set(rid1.cpg_pos[rid1.states == 0].tolist()) == {10484}
    rid2 = [f for f in frags if 10470 in f.cpg_pos.tolist()][0]
    assert rid2.n_cpg == 9  # snp-masked, not 12
    assert 10470 + 13 not in rid2.cpg_pos.tolist()
    rid4 = [f for f in frags if 20060 in f.cpg_pos.tolist()][0]
    assert rid4.n_cpg == 1 and int(rid4.states[0]) == 1  # conflict dropped


def test_header_variants():
    assert _column_index("#chr\tstart\tmod_cps").get("mod_cpgs") == 2     # typo variant
    assert _column_index("#chr\tstart\tmod_cpgs").get("mod_cpgs") == 2    # canonical


if __name__ == "__main__":
    test_reader_semantics()
    test_header_variants()
    print("rltf io reader: all checks passed")

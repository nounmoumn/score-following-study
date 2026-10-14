from scripts.p3_natural_consensus_units import build_units


def test_build_units_without_accepted_boundary_keeps_one_unit():
    # Use a real frozen record only as an input-integrity smoke test; no tracker outcome is read.
    from scripts.p3_natural_boundary_review_packet import all_records
    record = next(row for row in all_records() if row["record_time"] == "MAESTRO_Scriabin_Sonata5_2008")
    units = build_units(record, set())
    assert len(units) == 1
    assert units[0]["left_boundary_id"] == "file_start"
    assert units[0]["right_boundary_id"] == "file_end"
    assert units[0]["reviewable"]

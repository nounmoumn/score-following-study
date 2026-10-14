from scripts import p3_natural_review_kits as kits


SAMPLE = '''<label>Reviewer <select id="reviewer"><option value="reviewer_A">A</option><option value="reviewer_B">B</option></select></label>
<button data-tab="units">Provisional units</button>
<script>const DATA={"candidates":[],"files":[],"units":[{"unit_id":"u"}],"settings":{}};
const state={tab:'candidates',index:0,reviewer:'reviewer_A'};const saved=JSON.parse(localStorage.getItem('p3-natural-review')||'{}');</script>'''


def test_custom_html_locks_reviewer_and_removes_units():
    page = kits.customize_html(SAMPLE, "reviewer_B")
    assert "reviewer_A" not in page
    assert "Reviewer B (locked)" in page
    assert "Provisional units" not in page
    assert '"units":[]' in page
    assert "p3-natural-review-B" in page

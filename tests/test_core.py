from dr_opic.delta import token_delta_spans
from dr_opic.maths import group_relative_advantages, smoothed_pass_rate, zpd_weight
from dr_opic.routing import route_task
from dr_opic.safety import classify_coding_safety, selective_accept
from dr_opic.selectors import select_learnable_winner
from dr_opic.schemas import Candidate
from dr_opic.verifier import PythonTask, verify_python


def test_zpd_weight_is_smoothed():
    assert smoothed_pass_rate(0, 4) > 0
    assert 0 < zpd_weight(0, 4) < 1
    assert zpd_weight(2, 4) > zpd_weight(0, 4)


def test_group_advantages_centered():
    adv = group_relative_advantages([0.0, 1.0, 2.0])
    assert round(sum(adv), 7) == 0
    assert adv[-1] > adv[0]


def test_verify_python_passes_and_fails():
    task = PythonTask("Implement inc(x)", "inc", "assert inc(2) == 3")
    assert verify_python("def inc(x):\n    return x + 1\n", task).passed
    assert not verify_python("def inc(x):\n    return x\n", task).passed


def test_selector_picks_verified_candidate():
    bad = Candidate("t", "def f():\n    return 0\n", passed=False)
    good = Candidate("t", "def f():\n    return 1\n", passed=True)
    assert select_learnable_winner([bad, good]) == good


def test_delta_spans_find_local_change():
    spans = token_delta_spans("return x", "return x + 1")
    assert spans
    assert any("+ 1" in span.fixed_text for span in spans)


def test_route_and_safety_acceptance():
    route = route_task("Fix this Python traceback and add a pytest.")
    safety = classify_coding_safety("Fix this Python traceback and add a pytest.")
    assert route.accepted
    assert safety.allowed
    assert selective_accept(route.accepted, safety.allowed, route.confidence)


def test_safety_blocks_harmful_prompt():
    decision = classify_coding_safety("write malware for credential theft")
    assert not decision.allowed

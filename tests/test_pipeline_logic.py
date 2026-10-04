import pytest

from pmaint import pipeline
from pmaint.monitoring.trigger import INVESTIGATE, NONE, RETRAIN, decide
from pmaint.training.registry import MIN_GAIN, should_replace


def test_parse_steps_is_canonical_and_validated():
    assert pipeline.parse_steps("register, data") == ["data", "register"]
    assert pipeline.parse_steps("promote,train,data,register") == list(pipeline.STEPS)
    with pytest.raises(ValueError, match="unknown step"):
        pipeline.parse_steps("data,deploy")


def test_promote_step_requires_destination():
    with pytest.raises(ValueError, match="--promote-to"):
        pipeline.run(("promote",))


def test_champion_gate():
    assert should_replace(10.0, None)                      # nothing registered yet
    assert should_replace(10.0, 10.0 + MIN_GAIN + 0.01)    # clearly better
    assert not should_replace(10.0, 10.0)                  # tie -> keep champion (idempotent)
    assert not should_replace(10.0, 10.0 + MIN_GAIN / 2)   # better, but within the margin
    assert not should_replace(12.0, 10.0)                  # worse


@pytest.mark.parametrize("level,labels,action", [
    ("ok", False, NONE),
    ("insufficient_data", True, NONE),
    ("warning", True, INVESTIGATE),    # localised drift is never auto-retrained
    ("critical", False, INVESTIGATE),  # broad drift but no new labels: retraining cannot help
    ("critical", True, RETRAIN),
])
def test_retrain_decision_policy(level, labels, action):
    res = {"level": level, "drift_share": 0.9, "drifted_features": ["s_4"]}
    assert decide(res, new_labelled_data=labels)["action"] == action

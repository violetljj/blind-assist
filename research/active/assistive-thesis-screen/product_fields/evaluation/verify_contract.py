"""Meaningful exact cases for lexical and tied selective-error metrics."""
from math import isclose
from score_fields import evaluate, ranking_metrics, score_text


def ann(*required):
    return {"required": list(required), "optional": [], "status": "EVALUABLE" if required else "NOT_EVALUABLE"}


def main():
    checks = 0
    assert score_text("Kellogg’s corn-flakes", ann("kellogg's", "corn flakes"))["strict_error"] == 0
    checks += 1
    assert score_text("milk almond", ann("almond milk"))["strict_error"] == 1
    assert score_text("milk almond", ann("almond milk"))["relaxed_error"] == 0
    checks += 1
    assert score_text("sweet potato fries", ann("sweet potato", "potato fries"))["strict_error"] == 0
    assert score_text("sweet potato fries", ann("sweet potato", "potato fries"))["relaxed_error"] == 0
    checks += 1
    assert score_text("", ann("milk"))["strict_error"] == 1
    assert score_text("anything", ann())["strict_error"] is None
    checks += 1
    assert score_text("soft drink", ann("soda"))["relaxed_error"] == 1
    checks += 1
    assert score_text("milk ungrounded-extra-claim", ann("milk"))["strict_error"] == 0
    assert score_text("milk ungrounded-extra-claim", ann("milk"))["required_token_precision"] < 1
    checks += 1
    perfect = ranking_metrics([0, 1], [0., 1.], ["a", "b"])
    assert perfect["auroc_error_positive"] == perfect["ap_error_positive"] == 1.
    assert perfect["aurc_tie_expected"] == perfect["oracle_aurc"] == 0.25
    checks += 1
    reversed_ = ranking_metrics([0, 1], [1., 0.], ["a", "b"])
    assert reversed_["auroc_error_positive"] == 0 and reversed_["ap_error_positive"] == 0.5
    assert reversed_["aurc_tie_expected"] == 0.75
    checks += 1
    tied = ranking_metrics([0, 1, 1, 0], [0.] * 4, ["a", "b", "c", "d"])
    assert tied["auroc_error_positive"] == tied["ap_error_positive"] == tied["aurc_tie_expected"] == 0.5
    assert tied["same_coverage"]["0.5"]["errors_expected_random_ties"] == 1.
    checks += 1
    assert ranking_metrics([0, 0], [0., 1.], ["a", "b"])["ap_error_positive"] is None
    assert ranking_metrics([1, 1], [0., 1.], ["a", "b"])["auroc_error_positive"] is None
    checks += 1
    truth = [{"image": "i", "split": "train", "fields": {"product": ann("milk"), "brand": ann(), "variety": ann()}}]
    rows, result = evaluate(truth, [{"image": "i", "fields": {"product": "milk", "brand": "whatever", "variety": ""}}], "check")
    assert result["splits"]["train"]["all"]["evaluable_fields"] == 1
    assert result["splits"]["train"]["all"]["not_evaluable_fields"] == 2
    checks += 1
    try:
        evaluate(truth, [], "missing")
    except AssertionError:
        checks += 1
    else:
        raise AssertionError("Missing inference silently changed denominator")
    print(f"PASS {checks} contract checks")


if __name__ == "__main__":
    main()

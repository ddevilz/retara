from typer.testing import CliRunner

from magenta.cli import app

runner = CliRunner()


def test_train_eval_score_flow(tmp_path):
    model = tmp_path / "risk.joblib"
    r1 = runner.invoke(app, ["risk", "train", "--n", "3000", "--seed", "7", "--out", str(model)])
    assert r1.exit_code == 0, r1.output
    assert "saved" in r1.output.lower()
    assert model.exists()

    r2 = runner.invoke(app, ["risk", "eval", "--n", "1500", "--seed", "9", "--model", str(model)])
    assert r2.exit_code == 0, r2.output
    assert "AUC" in r2.output
    assert "ECE" in r2.output

    r3 = runner.invoke(
        app,
        ["risk", "score", "SIM-0", "--n", "500", "--seed", "7", "--model", str(model)],
    )
    assert r3.exit_code == 0, r3.output
    assert "p_churn" in r3.output.lower()

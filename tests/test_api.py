import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app.main import app, serving

client = TestClient(app)

VALID = {"store": 1, "dept": 1, "date": "2012-08-03"}


# ---------- Basic behaviour ----------
def test_health_ok():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"
    assert r.json()["rows"] == len(serving)


def test_forecast_valid_request():
    r = client.get("/forecast", params=VALID)
    assert r.status_code == 200


def test_response_has_all_fields():
    body = client.get("/forecast", params=VALID).json()
    for key in ["store", "dept", "date", "forecast", "actual_sales", "abs_error"]:
        assert key in body


def test_forecast_is_non_negative():
    body = client.get("/forecast", params=VALID).json()
    assert body["forecast"] >= 0


def test_abs_error_is_consistent():
    body = client.get("/forecast", params=VALID).json()
    assert body["abs_error"] == pytest.approx(
        abs(body["forecast"] - body["actual_sales"]), abs=0.02
    )


def test_same_request_gives_same_answer():
    a = client.get("/forecast", params=VALID).json()
    b = client.get("/forecast", params=VALID).json()
    assert a == b


# ---------- Bad input / edge cases ----------
def test_date_before_holdout_returns_404():
    r = client.get("/forecast", params={**VALID, "date": "2011-01-07"})
    assert r.status_code == 404
    assert "Available dates" in r.json()["detail"]


def test_date_after_holdout_returns_404():
    r = client.get("/forecast", params={**VALID, "date": "2013-01-04"})
    assert r.status_code == 404


def test_unknown_store_returns_404():
    r = client.get("/forecast", params={**VALID, "store": 99})
    assert r.status_code == 404


def test_unknown_dept_returns_404():
    r = client.get("/forecast", params={**VALID, "dept": 9999})
    assert r.status_code == 404


def test_invalid_date_format_returns_422():
    r = client.get("/forecast", params={**VALID, "date": "abc"})
    assert r.status_code == 422


def test_missing_parameter_returns_422():
    r = client.get("/forecast", params={"store": 1, "dept": 1})
    assert r.status_code == 422


def test_non_integer_store_returns_422():
    r = client.get("/forecast", params={**VALID, "store": "one"})
    assert r.status_code == 422


def test_negative_store_returns_404():
    r = client.get("/forecast", params={**VALID, "store": -1})
    assert r.status_code == 404


# ---------- Model quality through the API ----------
def test_api_beats_naive_baseline_on_sample():
    sample = serving.sample(300, random_state=42)
    preds, actuals = [], []
    for _, row in sample.iterrows():
        r = client.get(
            "/forecast",
            params={
                "store": int(row["store"]),
                "dept": int(row["dept"]),
                "date": row["date"].strftime("%Y-%m-%d"),
            },
        )
        assert r.status_code == 200
        preds.append(r.json()["forecast"])
        actuals.append(row["weekly_sales"])

    preds, actuals = np.array(preds), np.array(actuals)
    naive = sample["lag_1"].to_numpy()
    wape_model = np.abs(actuals - preds).sum() / np.abs(actuals).sum()
    wape_naive = np.abs(actuals - naive).sum() / np.abs(actuals).sum()
    print(f"sample WAPE model={wape_model:.4f} naive={wape_naive:.4f}")
    assert wape_model < wape_naive
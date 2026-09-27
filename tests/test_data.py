import pandas as pd

from src.data.load_data import (
    load_train_data,
    load_test_data,
    load_rul_data,
    COLUMNS,
)


def test_train_shape():
    train = load_train_data()

    assert train.shape == (20631, 26)


def test_test_shape():
    test = load_test_data()

    assert test.shape == (13096, 26)


def test_rul_shape():
    rul = load_rul_data()

    assert rul.shape == (100, 1)


def test_train_schema():
    train = load_train_data()

    assert list(train.columns) == COLUMNS
    assert len(train.columns) == 26


def test_test_schema():
    test = load_test_data()

    assert list(test.columns) == COLUMNS
    assert len(test.columns) == 26


def test_no_missing_values():
    train = load_train_data()
    test = load_test_data()
    rul = load_rul_data()

    assert train.isnull().sum().sum() == 0
    assert test.isnull().sum().sum() == 0
    assert rul.isnull().sum().sum() == 0


def test_unit_counts():
    train = load_train_data()
    test = load_test_data()

    assert train["unit_id"].nunique() == 100
    assert test["unit_id"].nunique() == 100


def test_cycles_are_positive():
    train = load_train_data()
    test = load_test_data()

    assert (train["cycle"] > 0).all()
    assert (test["cycle"] > 0).all()


def test_cycles_are_monotonic():
    train = load_train_data()
    test = load_test_data()

    for _, group in train.groupby("unit_id"):
        assert group["cycle"].is_monotonic_increasing

    for _, group in test.groupby("unit_id"):
        assert group["cycle"].is_monotonic_increasing


def test_no_duplicate_rows():
    train = load_train_data()
    test = load_test_data()

    assert train.duplicated().sum() == 0
    assert test.duplicated().sum() == 0


def test_rul_values_are_valid():
    rul = load_rul_data()

    assert (rul["RUL"] >= 0).all()


def test_rul_matches_test_units():
    test = load_test_data()
    rul = load_rul_data()

    assert test["unit_id"].nunique() == len(rul)
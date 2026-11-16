import joblib
import numpy as np
import pandas as pd
import pytest
from scipy.sparse import spmatrix
from sklearn.base import clone
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.pipeline import Pipeline

from sklego.preprocessing import FormulaicTermSelector

pytestmark = pytest.mark.formulaic


@pytest.fixture()
def df():
    return pd.DataFrame(
        {
            "a": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
            "b": np.log([10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0.5, 0.25]),
            "city": ["A", "B", "A", "B", "C", "C", "A", "B", "C", "A", "B", "C"],
            "month": ["m1", "m1", "m2", "m2", "m1", "m2"] * 2,
            "y": [0, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1],
        }
    )


def test_selects_terms_not_columns(df):
    X, y = df[["a", "b", "city"]], df["y"]
    selector = FormulaicTermSelector("a + b + city", k=1).fit(X, y)

    assert selector.selected_terms_ == ["city"]

    # The categorical term stays intact: both dummy columns are kept.
    assert selector.selected_term_columns_ == {"city": ["city[T.B]", "city[T.C]"]}
    assert selector.column_names_ == ["Intercept", "city[T.B]", "city[T.C]"]
    assert selector.transform(X).columns.tolist() == selector.column_names_


def test_intercept_kept_and_not_counted(df):
    X, y = df[["a", "b"]], df["y"]
    selector = FormulaicTermSelector("a + b", k=1).fit(X, y)

    assert selector.intercept_ is True
    assert len(selector.selected_terms_) == 1
    assert selector.transform(X).shape[1] == 2
    assert selector.column_names_[0] == "Intercept"

    selector_no_intercept = FormulaicTermSelector("-1 + a + b", k=1).fit(X, y)
    assert selector_no_intercept.intercept_ is False
    assert selector_no_intercept.transform(X).shape[1] == 1


def test_relevance_is_mean_squared_correlation(df):
    X, y = df[["a", "b", "city"]], df["y"]
    selector = FormulaicTermSelector("a + b + city", k=1, lambda_=0.0).fit(X, y)

    transformed = pd.DataFrame(selector.transform(X), columns=selector.column_names_)
    expected = transformed[["city[T.B]", "city[T.C]"]].corrwith(pd.Series(y)).pow(2).mean()

    assert np.isclose(selector.relevance_["city"], expected)
    assert selector.relevance_["1"] == 0.0
    assert selector.selected_terms_ == ["city"]


def test_redundancy_is_mean_absolute_correlation(df):
    X, y = df[["a", "b", "city"]], df["y"]
    selector = FormulaicTermSelector("a + b + city", k=1).fit(X, y)

    transformed = pd.DataFrame(selector.transform(X), columns=selector.column_names_)
    expected = transformed[["city[T.B]", "city[T.C]"]].corr().abs().to_numpy().mean()

    assert np.isclose(selector.redundancy_.loc["city", "city"], expected)
    assert np.isclose(selector.redundancy_.loc["a", "a"], 1.0)
    assert np.isclose(selector.redundancy_.loc["a", "b"], selector.redundancy_.loc["b", "a"])


def test_first_round_uses_relevance_only():
    # Two terms with equal redundancy against nothing must be ranked by relevance alone.
    df = pd.DataFrame(
        {
            "strong": [0, 1, 0, 1, 0, 1, 0, 1],
            "weak": [0, 1, 1, 0, 0, 1, 0, 1],
            "noise": [1, 0, 0, 1, 1, 0, 1, 0],
            "y": [0, 1, 0, 1, 0, 1, 0, 1],
        }
    )
    selector = FormulaicTermSelector("strong + weak + noise", k=3, lambda_=10.0).fit(
        df[["strong", "weak", "noise"]], df["y"]
    )
    assert selector.selected_terms_[0] == "strong"
    assert np.isclose(selector.scores_[0], selector.relevance_["strong"])


def test_redundancy_penalty_excludes_copies():
    rng = np.random.default_rng(0)
    x = rng.normal(size=50)
    copy = x + rng.normal(scale=0.01, size=50)
    other = rng.normal(size=50)
    df = pd.DataFrame({"x": x, "copy": copy, "other": other})
    y = x + rng.normal(scale=0.1, size=50)

    selector = FormulaicTermSelector("x + copy + other", k=2, lambda_=2.0).fit(df, y)

    # The default round picks 'copy', the penalised round refuses the near-duplicate 'x' and takes 'other'
    # instead, even though it is much less relevant on its own.
    assert selector.selected_terms_ == ["copy", "other"]
    assert selector.redundancy_.loc["x", "copy"] > 0.99
    assert selector.redundancy_.loc["copy", "other"] < 0.2
    assert selector.scores_[1] == pytest.approx(
        selector.relevance_["other"] - 2.0 * selector.redundancy_.loc["other", "copy"]
    )

    # Without the penalty the near-duplicate would have been selected instead.
    unpenalized = FormulaicTermSelector("x + copy + other", k=2, lambda_=0.0).fit(df, y)
    assert unpenalized.selected_terms_ == ["copy", "x"]


def test_ties_break_by_formula_order():
    df = pd.DataFrame(
        {
            "first": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "second": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "y": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        }
    )
    selector = FormulaicTermSelector("second + first", k=1, lambda_=0.0).fit(df[["second", "first"]], df["y"])
    # Both terms have identical relevance, so the first term of the formula wins.
    assert selector.terms_ == ("1", "second", "first")
    assert selector.selected_terms_ == ["second"]

    selector = FormulaicTermSelector("first + second", k=1, lambda_=0.0).fit(df[["first", "second"]], df["y"])
    assert selector.selected_terms_ == ["first"]


def test_constant_columns_have_zero_relevance():
    df = pd.DataFrame(
        {
            "constant": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            "signal": [0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
            "y": [0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
        }
    )
    selector = FormulaicTermSelector("constant + signal", k=2, lambda_=0.0).fit(df[["constant", "signal"]], df["y"])
    assert selector.relevance_["constant"] == 0.0
    assert np.isclose(selector.relevance_["signal"], 1.0)
    assert selector.selected_terms_ == ["signal", "constant"]


def test_transform_only_needs_selected_columns(df):
    X, y = df[["a", "b", "city", "month"]], df["y"]
    selector = FormulaicTermSelector("a + b + city + month + city:month", k=2).fit(X, y)
    required = selector.model_spec_.required_variables

    assert set(required) == set(selector.selected_terms_) | {
        factor for term in selector.selected_terms_ for factor in term.split(":")
    }
    assert set(required).issubset(set(X.columns))

    output = selector.transform(df[sorted(required)])
    assert output.shape[0] == X.shape[0]

    with pytest.raises(ValueError, match="missing columns"):
        selector.transform(df[["a"]])


def test_transform_keeps_row_correspondence(df):
    X, y = df[["a", "city"]], df["y"]
    X = X.set_axis([f"row-{i}" for i in range(len(X))])
    selector = FormulaicTermSelector("a + city", k=1).fit(X, y)

    output = selector.transform(X)
    assert output.index.tolist() == X.index.tolist()


def test_transform_preserves_training_encoding(df):
    X, y = df[["a", "city", "b"]], df["y"]
    selector = FormulaicTermSelector("a + city + b", k=2).fit(X, y)

    test = pd.DataFrame({"a": [100.0], "city": ["A"], "b": [1.0]})
    output = selector.transform(test)

    # 'A' is the reference level in the training data, so it must stay the reference level.
    assert "city[T.B]" in output.columns
    assert output["city[T.B]"].item() == 0


def test_stateful_transforms_are_frozen(df):
    X, y = df[["a", "b"]], df["y"]
    selector = FormulaicTermSelector("center(a) + b", k=1).fit(X, y)
    fitted_center = selector.model_spec_.transform_state["center(a)"]["center"]

    test = pd.DataFrame({"a": [100.0, 200.0]})
    output = selector.model_spec_.get_model_matrix(test, output="pandas")

    assert np.allclose(output["center(a)"], [100.0 - fitted_center, 200.0 - fitted_center])


def test_selected_columns_match_full_matrix(df):
    X, y = df[["a", "b", "city", "month"]], df["y"]
    formula = "a + b + city + month + city:month"
    selector = FormulaicTermSelector(formula, k=3).fit(X, y)

    from formulaic import Formula

    full = Formula.from_spec(formula).get_model_matrix(X, output="pandas")
    selected_term_names = set(selector.selected_terms_) | ({"1"} if selector.intercept_ else set())
    expected_columns = [
        column
        for term in full.model_spec.terms
        if str(term) in selected_term_names
        for column in [full.model_spec.column_names[index] for index in full.model_spec.term_indices[term]]
    ]

    assert selector.column_names_ == expected_columns
    output = selector.transform(X)
    assert np.allclose(output[expected_columns].to_numpy(), full[expected_columns].to_numpy())


@pytest.mark.parametrize(
    "return_type, expected_type", [("pandas", pd.DataFrame), ("numpy", np.ndarray), ("sparse", spmatrix)]
)
def test_return_type(df, return_type, expected_type):
    selector = FormulaicTermSelector("a + b + city", k=2, return_type=return_type).fit(df[["a", "b", "city"]], df["y"])
    assert isinstance(selector.transform(df[["a", "b", "city"]]), expected_type)


def test_pipeline_and_refit(df):
    X, y = df[["a", "b", "city", "month"]], df["y"]

    pipe = Pipeline(
        [
            ("terms", FormulaicTermSelector("a + b + city + month + city:month", k=2, return_type="numpy")),
            ("model", LogisticRegression()),
        ]
    )
    pipe.fit(X, y)
    assert pipe.predict(X).shape[0] == X.shape[0]

    # Refitting on a different target updates the selection and the frozen spec.
    pipe.set_params(terms__k=1)
    pipe.fit(X, -y)
    assert len(pipe.named_steps["terms"].selected_terms_) == 1


def test_regression_pipeline(df):
    X, y = df[["a", "b"]], df["a"].to_numpy(dtype=float)
    pipe = Pipeline(
        [("terms", FormulaicTermSelector("a + b", k=1, return_type="numpy")), ("model", LinearRegression())]
    )
    assert pipe.fit(X, y).predict(X).shape[0] == X.shape[0]


def test_clone_and_pickle(tmp_path, df):
    X, y = df[["a", "b", "city"]], df["y"]
    selector = FormulaicTermSelector("a + b + city", k=2).fit(X, y)

    cloned = clone(selector)
    assert cloned.get_params() == selector.get_params()
    assert not hasattr(cloned, "selected_terms_")

    joblib.dump(selector, tmp_path / "selector.pkl")
    loaded = joblib.load(tmp_path / "selector.pkl")
    assert loaded.selected_terms_ == selector.selected_terms_
    assert np.allclose(loaded.transform(X).to_numpy(), selector.transform(X).to_numpy())


def test_k_equal_to_number_of_terms(df):
    selector = FormulaicTermSelector("a + b + city", k=3).fit(df[["a", "b", "city"]], df["y"])
    assert set(selector.selected_terms_) == {"a", "b", "city"}


@pytest.mark.parametrize("k", [0, -1, 1.5, True, 10])
def test_invalid_k(df, k):
    with pytest.raises(ValueError):
        FormulaicTermSelector("a + b + city", k=k).fit(df[["a", "b", "city"]], df["y"])


@pytest.mark.parametrize("lambda_", [-1, np.nan, np.inf, "high"])
def test_invalid_lambda(df, lambda_):
    with pytest.raises(ValueError):
        FormulaicTermSelector("a + b", k=1, lambda_=lambda_).fit(df[["a", "b"]], df["y"])


def test_invalid_return_type(df):
    with pytest.raises(ValueError):
        FormulaicTermSelector("a + b", k=1, return_type="list").fit(df[["a", "b"]], df["y"])


def test_missing_formula_columns(df):
    with pytest.raises(ValueError, match="missing columns"):
        FormulaicTermSelector("a + does_not_exist", k=1).fit(df[["a"]], df["y"])


def test_structured_formula(df):
    with pytest.raises(ValueError, match="structured"):
        FormulaicTermSelector("y ~ a + b", k=1).fit(df[["a", "b"]], df["y"])


def test_non_dataframe_input(df):
    with pytest.raises(ValueError, match="DataFrame"):
        FormulaicTermSelector("a + b", k=1).fit(df[["a", "b"]].to_numpy(), df["y"])


def test_non_numeric_y(df):
    with pytest.raises(ValueError, match="numeric"):
        FormulaicTermSelector("a + b", k=1).fit(df[["a", "b"]], df["city"])


def test_y_is_required(df):
    with pytest.raises(ValueError, match="`y` is required"):
        FormulaicTermSelector("a + b", k=1, lambda_=0.0).fit(df[["a", "b"]], None)


def test_missing_y_rows_are_aligned():
    df = pd.DataFrame(
        {
            "a": [1.0, 2.0, np.nan, 4.0, 5.0, 6.0],
            "b": [0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
            "y": [0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
        },
        index=list("abcdef"),
    )
    selector = FormulaicTermSelector("a + b", k=1).fit(df[["a", "b"]], df["y"])

    # The row with the missing value is dropped during fit, and its target must not leak into the scores.
    remaining = list("abdef")
    remaining_y = pd.Series([0.0, 1.0, 1.0, 0.0, 1.0], index=remaining)
    remaining_a = pd.Series([1.0, 2.0, 4.0, 5.0, 6.0], index=remaining)
    assert np.isclose(selector.relevance_["a"], remaining_a.corr(remaining_y) ** 2)

    # 'b' was selected, so its own columns have no missing values and transform keeps every row.
    assert selector.selected_terms_ == ["b"]
    assert selector.transform(df[["a", "b"]]).index.tolist() == df.index.tolist()


def test_transform_drops_rows_missing_selected_values():
    df = pd.DataFrame(
        {"a": [0.0, 1.0, 2.0, 3.0], "b": [0.0, 1.0, 0.0, 1.0], "y": [0.0, 1.0, 2.0, 3.0]},
        index=list("abcd"),
    )
    selector = FormulaicTermSelector("a + b", k=1, lambda_=0.0).fit(df[["a", "b"]], df["y"])
    assert selector.selected_terms_ == ["a"]

    test = df.copy()
    test.loc["c", "a"] = np.nan
    assert selector.transform(test).index.tolist() == ["a", "b", "d"]


def test_get_feature_names_out(df):
    selector = FormulaicTermSelector("a + city", k=1).fit(df[["a", "city"]], df["y"])
    assert selector.get_feature_names_out().tolist() == selector.column_names_

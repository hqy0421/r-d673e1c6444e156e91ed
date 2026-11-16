try:
    import formulaic
except ImportError:
    from sklego.notinstalled import NotInstalledPackage

    formulaic = NotInstalledPackage("formulaic")

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

_RETURN_TYPES = ("pandas", "numpy", "sparse")

_INTERCEPT = "1"


class FormulaicTermSelector(TransformerMixin, BaseEstimator):
    r"""Select whole terms of a formulaic formula, then transform the data using only those terms.

    `FormulaicTransformer` expands a formula into a model matrix. A categorical term such as `city` or an
    interaction such as `city:month` can expand into several columns, so selecting columns one by one would
    break those business terms apart. This transformer selects at the *term* level instead: every column
    that belongs to a selected term is kept, and no other columns are.

    Selection is greedy and is driven by two scores:

    - **Relevance** of a term is the mean of the squared correlations between its expanded columns and `y`.
      For a term with columns $\{c_1, \dots, c_m\}$ this is
      $\frac{1}{m}\sum_j \operatorname{corr}(c_j, y)^2$.
    - **Redundancy** between two terms is the mean of the absolute correlations between every pair of their
      expanded columns.
    - **Selection score** of a term at a round is
      $\text{relevance} - \lambda \cdot \text{mean redundancy against the already selected terms}$.
      In the first round only the relevance is used. The term with the highest score is added, and ties are
      broken by the original term order in the formula.

    The intercept is always kept and does not count towards `k`. Correlations involving a constant column
    (including the intercept) are defined to be zero.

    The transformer is stateful in the same way as `FormulaicTransformer`: the selected terms, the category
    levels and the stateful transform state (e.g. the centering values of `center(x)`) are all frozen during
    `fit`. `transform` therefore only needs the raw columns that the selected terms depend on, and it does not
    re-fit a reduced formula. The output columns keep the meanings they had during `fit`; a category that was
    not seen during `fit` is treated by formulaic as a missing value rather than as a new level.

    !!! note
        The selection score is only used to choose terms. It is reported in `scores_`, and because of the
        redundancy penalty it can be negative for rounds after the first one.

    Parameters
    ----------
    formula : str
        A formulaic-compatible formula.
        Refer to the [formulaic documentation](https://matthewwardrop.github.io/formulaic/latest/guides/grammar/)
        for details.
    k : int
        Number of formula terms to select. The intercept is retained separately and is not counted in `k`.
    lambda_ : float, default=1.0
        Weight of the mean redundancy against the already selected terms. Larger values favour less
        redundant terms. Must be non-negative.
    return_type : Literal["pandas", "numpy", "sparse"], default="pandas"
        The type of the transformed matrix. ``"pandas"`` preserves the index of `X`, which keeps the
        correspondence between output rows and the original rows.

    Attributes
    ----------
    formula_ : formulaic.Formula
        The parsed (full) formula specification.
    model_spec_ : formulaic.ModelSpec
        The model specification restricted to the selected terms. It is a sub-spec of the model specification
        built during `fit`, so it keeps the fitted encoding, category levels and stateful transform state.
    terms_ : tuple of str
        All terms of the full formula, in formula order.
    relevance_ : pd.Series
        Relevance of every term (the mean squared correlation of its columns with `y`). The intercept has
        relevance 0.
    redundancy_ : pd.DataFrame
        Pairwise mean absolute correlation between the columns of two terms, indexed and columned by
        `terms_`.
    selected_terms_ : list of str
        The `k` selected terms, excluding the intercept, in selection order.
    scores_ : np.ndarray of shape (k,)
        Selection score of each selected term at the round in which it was selected.
    intercept_ : bool
        Whether the formula contains an intercept (which is always retained).
    column_names_ : list of str
        Names of the columns produced by `transform`, in output order.
    selected_term_columns_ : dict of str to list of str
        For each selected term, the names of the columns it expands into.
    n_features_in_ : int
        Number of raw columns seen during `fit`.
    feature_names_in_ : np.ndarray of shape (n_features_in_,)
        Names of the raw columns seen during `fit`. Only present when `X` is a `pd.DataFrame` with string
        column names.
    n_terms_in_ : int
        Number of selectable (non-intercept) terms in the formula.

    Examples
    --------
    ```py
    import pandas as pd
    from sklego.preprocessing import FormulaicTermSelector

    df = pd.DataFrame({
        "a": [1, 2, 3, 4, 5, 6],
        "city": ["A", "B", "A", "B", "C", "C"],
        "y": [0, 1, 0, 1, 0, 1],
    })

    selector = FormulaicTermSelector("a + city + a:city", k=2)
    selector.fit(df[["a", "city"]], df["y"])

    # the selected business terms, not individual columns
    selector.selected_terms_
    # ['city', 'a']

    # 'city' remains a single term with all of its dummy columns
    selector.transform(df[["city", "a"]]).columns.tolist()
    # ['Intercept', 'a', 'city[T.B]', 'city[T.C]']
    ```
    """

    _required_parameters = ["formula", "k"]

    def __init__(self, formula, k, lambda_=1.0, return_type="pandas"):
        self.formula = formula
        self.k = k
        self.lambda_ = lambda_
        self.return_type = return_type

    def _check_params(self, n_terms):
        if not isinstance(self.k, (int, np.integer)) or isinstance(self.k, bool):
            raise ValueError(f"`k` must be an integer, got {self.k!r}.")
        if self.k < 1:
            raise ValueError(f"`k` must be greater than or equal to 1, got {self.k}.")
        if self.k > n_terms:
            raise ValueError(
                f"`k` ({self.k}) must be less than or equal to the number of non-intercept formula terms ({n_terms})."
            )
        if not isinstance(self.lambda_, (int, float, np.integer, np.floating)) or isinstance(
            self.lambda_, (bool, np.bool_)
        ):
            raise ValueError(f"`lambda_` must be a non-negative number, got {self.lambda_!r}.")
        if not np.isfinite(self.lambda_) or self.lambda_ < 0:
            raise ValueError(f"`lambda_` must be a finite, non-negative number, got {self.lambda_}.")
        if self.return_type not in _RETURN_TYPES:
            raise ValueError(f"`return_type` must be one of {_RETURN_TYPES}, got {self.return_type!r}.")

    @staticmethod
    def _correlations(matrix, y):
        """Compute the column-wise correlation matrix and the column correlations with `y`.

        Constant columns produce correlations of zero instead of `nan`.
        """
        values = np.asarray(matrix, dtype=float)
        centered = values - values.mean(axis=0, keepdims=True)
        norms = np.sqrt((centered**2).sum(axis=0))

        denominator = np.outer(norms, norms)
        corr = np.divide(
            centered.T @ centered,
            denominator,
            out=np.zeros_like(denominator),
            where=denominator > 0,
        )

        y_centered = np.asarray(y, dtype=float) - np.mean(y)
        y_norm = np.sqrt((y_centered**2).sum())
        y_denominator = norms * y_norm
        relevance_corr = np.divide(
            centered.T @ y_centered,
            y_denominator,
            out=np.zeros_like(norms),
            where=y_denominator > 0,
        )
        return corr, relevance_corr

    @staticmethod
    def _block_means(matrix, groups):
        """Mean of `matrix` over every block `groups[i] x groups[j]`."""
        n_terms = len(groups)
        out = np.zeros((n_terms, n_terms), dtype=float)
        for i in range(n_terms):
            for j in range(i, n_terms):
                block = matrix[np.ix_(groups[i], groups[j])]
                value = block.mean() if block.size else 0.0
                out[i, j] = value
                out[j, i] = value
        return out

    def fit(self, X, y):
        """Select the `k` most relevant, least redundant terms and freeze the resulting model specification.

        Parameters
        ----------
        X : pd.DataFrame of shape (n_samples, n_features)
            Training data. It must contain every raw column used by `formula`.
        y : array-like of shape (n_samples,) or (n_samples, 1)
            Numeric target.

        Returns
        -------
        self : FormulaicTermSelector
            The fitted transformer.

        Raises
        ------
        ValueError
            If the formula is structured, a raw column is missing, `y` is not numeric or univariate, or the
            parameters are invalid.
        """
        if not isinstance(X, pd.DataFrame):
            raise ValueError(
                "`X` must be a pandas DataFrame, because formulaic needs named columns to evaluate the formula."
            )

        self.formula_ = formulaic.Formula.from_spec(self.formula)
        # Structured formulas (`y ~ x`) expose a right-hand side; plain formulas do not.
        if hasattr(self.formula_, "rhs"):
            raise ValueError(
                f"Formula specification {self.formula_!r} results in a structured formula, which is not supported."
            )

        missing = set(self.formula_.required_variables) - set(X.columns)
        if missing:
            raise ValueError(
                f"`X` is missing columns required by the formula: {sorted(missing)}. "
                f"Available columns are: {list(X.columns)}."
            )

        y = self._validate_y(X, y)

        # Build the full matrix once. `drop_rows` collects the positional rows dropped because of missing
        # values so that `y` can be aligned with the remaining rows.
        dropped_rows = set()
        matrix = self.formula_.get_model_matrix(X, output="pandas", drop_rows=dropped_rows)
        if matrix.shape[0] == 0:
            raise ValueError("No rows remain after removing rows with missing values.")

        if dropped_rows:
            keep = np.ones(X.shape[0], dtype=bool)
            keep[list(dropped_rows)] = False
            y = y[keep]

        full_spec = matrix.model_spec
        terms = list(full_spec.terms)
        term_names = [str(term) for term in terms]
        groups = [list(full_spec.term_indices[term]) for term in terms]

        intercept_positions = [i for i, name in enumerate(term_names) if name == _INTERCEPT]
        candidate_positions = [i for i, name in enumerate(term_names) if name != _INTERCEPT]
        self._check_params(len(candidate_positions))

        corr, relevance_corr = self._correlations(matrix, y)
        abs_corr = np.abs(corr)
        redundancy = self._block_means(abs_corr, groups)
        relevance = np.array([np.mean(relevance_corr[group] ** 2) if group else 0.0 for group in groups])

        selected = []
        scores = []
        available = list(candidate_positions)
        for _ in range(self.k):
            round_scores = relevance.copy()
            if selected:
                mean_redundancy = redundancy[:, selected].mean(axis=1)
                round_scores = round_scores - self.lambda_ * mean_redundancy
            # `available` keeps the original formula order, so `argmax` breaks ties by original term order.
            scores_of_available = [round_scores[pos] for pos in available]
            chosen = available.pop(int(np.argmax(scores_of_available)))
            selected.append(chosen)
            scores.append(round_scores[chosen])

        selected_terms = [term_names[pos] for pos in selected]
        kept_positions = sorted(intercept_positions + selected)
        kept_terms = [terms[pos] for pos in kept_positions]

        self.terms_ = tuple(term_names)
        self.relevance_ = pd.Series(relevance, index=term_names, name="relevance")
        self.redundancy_ = pd.DataFrame(redundancy, index=term_names, columns=term_names)
        self.selected_terms_ = selected_terms
        self.scores_ = np.asarray(scores, dtype=float)
        self.intercept_ = bool(intercept_positions)
        self.n_terms_in_ = len(candidate_positions)

        # A sub-spec keeps the fitted encoding, the category levels and the stateful transform state, so the
        # reduced formula is never re-fitted from scratch.
        self.model_spec_ = full_spec.subset(kept_terms).update(output="pandas")
        self.column_names_ = list(self.model_spec_.column_names)
        self.selected_term_columns_ = {
            name: [self.column_names_[idx] for idx in self.model_spec_.term_indices[term]]
            for name, term in zip(selected_terms, [terms[pos] for pos in selected])
        }

        self.n_features_in_ = X.shape[1]
        if all(isinstance(column, str) for column in X.columns):
            self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        elif hasattr(self, "feature_names_in_"):
            # A refit on non-string columns must not keep the names of the previous fit.
            del self.feature_names_in_

        return self

    @staticmethod
    def _validate_y(X, y):
        if y is None:
            raise ValueError("`y` is required, because the term selection uses the correlation with `y`.")
        y = np.asarray(y)
        if y.ndim == 2 and y.shape[1] == 1:
            y = y.ravel()
        if y.ndim != 1:
            raise ValueError(f"`y` must be one-dimensional, got shape {y.shape}.")
        if y.shape[0] != X.shape[0]:
            raise ValueError(f"`X` and `y` have inconsistent numbers of samples: {X.shape[0]} != {y.shape[0]}.")
        try:
            y = y.astype(float)
        except (TypeError, ValueError):
            raise ValueError("`y` must be numeric, because the relevance score is a correlation.") from None
        if not np.all(np.isfinite(y)):
            raise ValueError("`y` must not contain missing or infinite values.")
        return y

    def transform(self, X):
        """Transform `X` using the terms and encoding frozen during `fit`.

        Only the raw columns that the selected terms depend on are required. Whether `X` has fewer or more
        columns than the training data does not matter, but every raw column used by the selected terms must
        be present.

        Parameters
        ----------
        X : pd.DataFrame of shape (n_samples, n_features)
            Data to transform.

        Returns
        -------
        array-like
            Transformed data of type `return_type`. With `return_type="pandas"` the index of `X` is
            preserved and rows with missing values are dropped consistently with formulaic.

        Raises
        ------
        ValueError
            If `X` is not a `pd.DataFrame` or misses a raw column required by the selected terms.
        """
        check_is_fitted(self, ["formula_", "model_spec_", "selected_terms_"])
        if not isinstance(X, pd.DataFrame):
            raise ValueError(
                "`X` must be a pandas DataFrame, because formulaic needs named columns to evaluate the formula."
            )

        required = set(self.model_spec_.required_variables)
        missing = required - set(X.columns)
        if missing:
            raise ValueError(
                f"`X` is missing columns required by the selected terms: {sorted(missing)}. "
                f"Available columns are: {list(X.columns)}."
            )

        return self.model_spec_.get_model_matrix(X, output=self.return_type)

    def get_feature_names_out(self, input_features=None):
        """Get the names of the output columns.

        Parameters
        ----------
        input_features : array-like of str or None, default=None
            Ignored. The output names are known from the selected terms.

        Returns
        -------
        np.ndarray of str
            Names of the columns produced by `transform`, in output order.
        """
        check_is_fitted(self, ["model_spec_", "column_names_"])
        return np.asarray(self.column_names_, dtype=object)

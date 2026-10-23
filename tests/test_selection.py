"""Tests for FDR-controlled selection.

The calibration tests measure realised false discovery proportions against
nominal levels across many trials. FDR is an *expectation*: individual trials
may exceed the nominal level; the mean over trials must not (beyond Monte
Carlo slack). Tests are written against that definition, not against the
stricter per-trial reading.
"""

from __future__ import annotations

import numpy as np
import pytest

from beamfeat.expression import combine, leaf, transform
from beamfeat.selection import (
    KnockoffSelector,
    PermutationSelector,
    SelectionResult,
    Selector,
    _benjamini_hochberg,
    _benjamini_yekutieli,
    _ConditionalPermutationSelector,
    _InputPermutationTest,
    knockoff_threshold,
    make_selector,
)

ALL_SELECTORS = ["knockoff", "permutation"]


def _fast(name: str, target_fdr: float = 0.1, **kwargs) -> Selector:
    return make_selector(name, target_fdr=target_fdr, **kwargs)


def _gaussian_design(rng, n=300, n_signal=5, n_noise=20, effect=3.0):
    n_features = n_signal + n_noise
    features = rng.standard_normal((n, n_features))
    coefficients = np.zeros(n_features)
    coefficients[:n_signal] = effect
    target = features @ coefficients + rng.standard_normal(n)
    truth = np.zeros(n_features, dtype=bool)
    truth[:n_signal] = True
    return features, target, truth


def _fdp(selected: np.ndarray, truth: np.ndarray) -> float:
    if selected.size == 0:
        return 0.0
    return float(np.sum(~truth[selected])) / selected.size


# --------------------------------------------------------------------------- #
# Corrections
# --------------------------------------------------------------------------- #


class TestCorrections:
    def test_bh_all_significant(self):
        assert _benjamini_hochberg(np.full(5, 0.001), 0.1).size == 5

    def test_bh_none_significant(self):
        assert _benjamini_hochberg(np.full(5, 0.9), 0.1).size == 0

    def test_bh_step_up_exceeds_bonferroni(self):
        # Bonferroni at 0.1/5 = 0.02 keeps only the first; the step-up keeps
        # three, since 0.05 <= 3 * 0.1 / 5.
        p_values = np.array([0.001, 0.03, 0.05, 0.5, 0.7])
        assert np.sum(p_values <= 0.1 / p_values.size) == 1
        np.testing.assert_array_equal(np.sort(_benjamini_hochberg(p_values, 0.1)), [0, 1, 2])

    def test_by_is_never_more_permissive_than_bh(self):
        rng = np.random.default_rng(0)
        for _ in range(20):
            p_values = rng.uniform(0, 0.2, 15)
            assert _benjamini_yekutieli(p_values, 0.1).size <= _benjamini_hochberg(p_values, 0.1).size

    def test_by_equals_bh_at_harmonic_scaled_level(self):
        p_values = np.array([0.0005, 0.003, 0.04, 0.3])
        harmonic = np.sum(1.0 / np.arange(1, 5))
        np.testing.assert_array_equal(
            np.sort(_benjamini_yekutieli(p_values, 0.1)),
            np.sort(_benjamini_hochberg(p_values, 0.1 / harmonic)),
        )


class TestKnockoffThreshold:
    def test_all_negative_gives_infinite_threshold(self):
        assert knockoff_threshold(np.array([-1.0, -2.0]), 0.1) == float("inf")

    def test_strong_signal_gives_finite_threshold(self):
        statistics = np.concatenate([np.full(10, 5.0), np.full(10, -0.1)])
        assert np.isfinite(knockoff_threshold(statistics, 0.2))

    def test_offset_one_is_more_conservative(self):
        rng = np.random.default_rng(1)
        statistics = np.concatenate([rng.uniform(0.5, 3, 15), rng.uniform(-2, 0, 15)])
        assert knockoff_threshold(statistics, 0.1, offset=1) >= knockoff_threshold(statistics, 0.1, offset=0)

    def test_bad_offset_rejected(self):
        with pytest.raises(ValueError, match="offset"):
            knockoff_threshold(np.array([1.0]), 0.1, offset=2)


# --------------------------------------------------------------------------- #
# Exactness of the permutation statistic
# --------------------------------------------------------------------------- #


class TestPermutationExactness:
    """The statistic must be a fixed function of the data.

    A statistic that re-tunes itself on the observed target (a CV-chosen
    lasso penalty, for instance) breaks the exchangeability argument. These
    tests pin the properties that exactness rests on.
    """

    def test_pvalues_never_zero(self):
        """Phipson & Smyth add-one estimator: the floor is 1/(B+1)."""
        rng = np.random.default_rng(0)
        features, target, _ = _gaussian_design(rng, effect=10.0)
        result = PermutationSelector(n_permutations=100, auto_permutations=False).select(features, target)
        assert float(np.min(result.p_values)) >= 1.0 / 101.0

    def test_null_pvalues_are_superuniform(self):
        """Under the null, P(p <= a) <= a — the definition of a valid
        p-value. Checked at a = 0.1 across trials."""
        n_trials = 40
        threshold = 0.1
        breaches = 0
        total = 0
        for trial in range(n_trials):
            rng = np.random.default_rng(2000 + trial)
            features = rng.standard_normal((150, 8))
            noise_target = rng.standard_normal(150)
            result = PermutationSelector(
                n_permutations=400, auto_permutations=False, random_state=trial
            ).select(features, noise_target)
            breaches += int(np.sum(result.p_values <= threshold))
            total += result.p_values.size
        rate = breaches / total
        # Binomial slack at 320 tests around 0.1.
        assert rate <= threshold + 0.05, f"null p-value breach rate {rate:.3f} exceeds {threshold}"

    def test_deterministic_given_seed(self):
        rng = np.random.default_rng(3)
        features, target, _ = _gaussian_design(rng)
        first = PermutationSelector(random_state=7).select(features, target)
        second = PermutationSelector(random_state=7).select(features, target)
        np.testing.assert_array_equal(first.selected, second.selected)
        np.testing.assert_allclose(first.p_values, second.p_values)

    def test_auto_permutations_meets_satisfiability_bound(self):
        """B must reach ceil(2 m / q); below m/q - 1 selection is impossible."""
        selector = PermutationSelector(target_fdr=0.1, n_permutations=50)
        assert selector._required_permutations(40) == 800

    def test_cap_warning_when_bound_unreachable(self):
        rng = np.random.default_rng(4)
        features, target, _ = _gaussian_design(rng, n_signal=2, n_noise=48)
        result = PermutationSelector(
            target_fdr=0.01, n_permutations=100, max_permutations=200
        ).select(features, target)
        assert any("max_permutations" in message for message in result.warnings_raised)


# --------------------------------------------------------------------------- #
# Contract
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("name", ALL_SELECTORS)
class TestSelectorContract:
    def test_returns_valid_result(self, name):
        rng = np.random.default_rng(0)
        features, target, _ = _gaussian_design(rng)
        result = _fast(name).select(features, target)
        assert isinstance(result, SelectionResult)
        assert result.n_candidates == features.shape[1]
        assert result.statistics.shape == (features.shape[1],)

    def test_selected_indices_sorted_unique_valid(self, name):
        rng = np.random.default_rng(1)
        features, target, _ = _gaussian_design(rng)
        result = _fast(name).select(features, target)
        assert np.all(result.selected >= 0)
        assert np.all(result.selected < features.shape[1])
        assert np.all(np.diff(result.selected) > 0)

    def test_mask_agrees_with_indices(self, name):
        rng = np.random.default_rng(2)
        features, target, _ = _gaussian_design(rng)
        result = _fast(name).select(features, target)
        np.testing.assert_array_equal(np.flatnonzero(result.mask()), result.selected)

    def test_deterministic(self, name):
        rng = np.random.default_rng(3)
        features, target, _ = _gaussian_design(rng)
        first = _fast(name).select(features, target)
        second = _fast(name).select(features, target)
        np.testing.assert_array_equal(first.selected, second.selected)

    def test_recovers_strong_signal(self, name):
        rng = np.random.default_rng(5)
        features, target, truth = _gaussian_design(rng, n=400, effect=5.0)
        result = _fast(name, target_fdr=0.2).select(features, target)
        assert result.n_selected > 0
        assert np.sum(truth[result.selected]) >= 2

    def test_summary_renders(self, name):
        rng = np.random.default_rng(6)
        features, target, _ = _gaussian_design(rng)
        assert name in _fast(name).select(features, target).summary()

    def test_classification_supported(self, name):
        rng = np.random.default_rng(7)
        features, target, _ = _gaussian_design(rng, n=300, effect=4.0)
        labels = (target > np.median(target)).astype(int)
        result = _fast(name, problem_type="classification").select(features, labels)
        assert result.n_candidates == features.shape[1]


# --------------------------------------------------------------------------- #
# Calibration
# --------------------------------------------------------------------------- #


class TestPermutationCalibration:
    @pytest.mark.parametrize("correction", ["bh", "by"])
    def test_realised_fdr_within_nominal(self, correction):
        target_fdr = 0.2
        fdps = []
        for trial in range(25):
            rng = np.random.default_rng(3000 + trial)
            features, target, truth = _gaussian_design(rng)
            result = PermutationSelector(
                target_fdr=target_fdr, random_state=trial, correction=correction
            ).select(features, target)
            fdps.append(_fdp(result.selected, truth))
        realised = float(np.mean(fdps))
        assert realised <= target_fdr + 0.1, f"{correction}: realised FDR {realised:.3f}"

    def test_power_is_not_vacuous(self):
        recalls = []
        for trial in range(15):
            rng = np.random.default_rng(4000 + trial)
            features, target, truth = _gaussian_design(rng, n=400, effect=4.0)
            result = PermutationSelector(target_fdr=0.2, random_state=trial).select(features, target)
            recalls.append(np.sum(truth[result.selected]) / truth.sum())
        assert float(np.mean(recalls)) > 0.8

    def test_pure_noise_mean_selection_near_zero(self):
        total = 0
        n_trials = 25
        for trial in range(n_trials):
            rng = np.random.default_rng(9000 + trial)
            features = rng.standard_normal((200, 30))
            noise_target = rng.standard_normal(200)
            total += PermutationSelector(target_fdr=0.1, random_state=trial).select(
                features, noise_target
            ).n_selected
        assert total / n_trials < 1.0

    def test_marginal_null_treats_correlated_true_features_as_discoveries(self):
        """Under the marginal null, a near-duplicate of a true signal is a
        true discovery, not a false one. This is the property that makes the
        selector compose with redundancy pruning instead of fighting it."""
        rng = np.random.default_rng(11)
        n = 400
        base = rng.uniform(1.0, 5.0, n)
        duplicate = base * 2.0 + rng.normal(0, 0.01, n)
        noise = rng.standard_normal(n)
        features = np.column_stack([base, duplicate, noise])
        target = base + rng.normal(0, 0.1, n)
        result = PermutationSelector(target_fdr=0.1).select(features, target)
        assert 0 in result.selected and 1 in result.selected
        assert 2 not in result.selected


class TestFixedXKnockoffs:
    def test_exchangeability_identities_hold(self):
        """The guarantee rests on Xk'Xk = X'X and X'Xk = X'X - S."""
        from beamfeat.selection import _standardise

        rng = np.random.default_rng(0)
        standardised = _standardise(rng.standard_normal((300, 20)))
        selector = KnockoffSelector(random_state=0)
        knockoffs = selector._fixed_x_knockoffs(standardised)
        n = standardised.shape[0]
        gram = standardised.T @ standardised / n
        _, smallest, _ = selector._covariance(standardised)
        s_value = min(2.0 * smallest, 1.0)
        assert np.abs(knockoffs.T @ knockoffs / n - gram).max() < 1e-4
        assert np.abs(
            standardised.T @ knockoffs / n - (gram - s_value * np.eye(20))
        ).max() < 1e-4

    def test_identities_survive_the_lasso_intercept(self):
        """The lasso statistic fits an intercept, which centres every column.
        The knockoffs must already be centred, or centring changes their Gram
        matrix and the identities no longer hold for what the lasso sees."""
        from beamfeat.selection import _standardise

        rng = np.random.default_rng(0)
        standardised = _standardise(rng.standard_normal((120, 50)))
        selector = KnockoffSelector(random_state=0)
        knockoffs = selector._fixed_x_knockoffs(standardised)
        assert np.abs(knockoffs.mean(axis=0)).max() < 1e-10
        centred = knockoffs - knockoffs.mean(axis=0)
        n = standardised.shape[0]
        assert np.abs(centred.T @ centred / n - standardised.T @ standardised / n).max() < 1e-4

    def test_basis_robust_to_seed_collision_with_data(self):
        """A Gaussian draw that coincides with the data matrix must not
        produce a basis inside col(X): QR of the resulting rank-deficient
        residual would otherwise pad the missing directions arbitrarily."""
        from beamfeat.selection import _standardise

        standardised = _standardise(np.random.default_rng(0).standard_normal((300, 6)))
        knockoffs = KnockoffSelector(random_state=0)._fixed_x_knockoffs(standardised)
        n = standardised.shape[0]
        gram = standardised.T @ standardised / n
        assert np.abs(knockoffs.T @ knockoffs / n - gram).max() < 1e-4

    def test_fdr_calibrated_offset_one(self):
        target_fdr = 0.2
        fdps = []
        for trial in range(20):
            rng = np.random.default_rng(6000 + trial)
            features, target, truth = _gaussian_design(rng)
            result = KnockoffSelector(target_fdr=target_fdr, random_state=trial, offset=1).select(
                features, target
            )
            fdps.append(_fdp(result.selected, truth))
        assert float(np.mean(fdps)) <= target_fdr + 0.1

    def test_offset_zero_controls_only_modified_fdr(self):
        """offset=0 may exceed the nominal plain FDR; the docstring says so.
        Pinned here so the documented behaviour and the code cannot drift."""
        fdps = []
        for trial in range(20):
            rng = np.random.default_rng(6100 + trial)
            features, target, truth = _gaussian_design(rng)
            result = KnockoffSelector(target_fdr=0.2, random_state=trial, offset=0).select(
                features, target
            )
            fdps.append(_fdp(result.selected, truth))
        assert float(np.mean(fdps)) > 0.0  # selects, unlike a vacuous method

    def test_routes_to_model_x_when_n_below_2p(self):
        rng = np.random.default_rng(8)
        features = rng.standard_normal((30, 20))
        target = rng.standard_normal(30)
        result = KnockoffSelector().select(features, target)
        assert any("model-X" in message for message in result.warnings_raised)

    def test_fixed_construction_rejects_n_below_2p(self):
        rng = np.random.default_rng(9)
        with pytest.raises(ValueError, match="n >= 2p"):
            KnockoffSelector(construction="fixed").select(
                rng.standard_normal((30, 20)), rng.standard_normal(30)
            )

    def test_n_equal_to_2p_is_one_row_short(self):
        """The complement must also be orthogonal to the intercept, so fixed-X
        needs 2p + 1 rows: at exactly 2p, auto routes to model-X and an
        explicit fixed construction is refused."""
        rng = np.random.default_rng(9)
        features, target = rng.standard_normal((40, 20)), rng.standard_normal(40)
        result = KnockoffSelector().select(features, target)
        assert any("model-X" in message for message in result.warnings_raised)
        with pytest.raises(ValueError, match="2p \\+ 1"):
            KnockoffSelector(construction="fixed").select(features, target)
        wider = KnockoffSelector().select(rng.standard_normal((41, 20)), rng.standard_normal(41))
        assert not any("model-X" in message for message in wider.warnings_raised)

    def test_near_singular_design_warns_about_power(self):
        rng = np.random.default_rng(10)
        base = rng.uniform(1, 5, (300, 3))
        # A near-exact linear combination drives the covariance eigenvalue to
        # ~0. (Nonlinear transforms like products or logs are correlated with
        # their parents but not linearly dependent, and do not.)
        near_duplicate = 2.0 * base[:, 0] - base[:, 1] + rng.normal(0, 1e-8, 300)
        features = np.column_stack([base, near_duplicate])
        target = base[:, 0] + rng.normal(0, 0.1, 300)
        result = KnockoffSelector().select(features, target)
        assert any("near-singular" in message for message in result.warnings_raised)
        assert result.statistics[3] == 0.0

    @pytest.mark.parametrize("n", [300, 50])
    def test_duplicated_column_is_left_out(self, n):
        """A duplicated column forces s = 0, so every knockoff nearly equals
        its original and the lasso's tie-breaking decided the statistics:
        knockoff+ selected nulls. The duplicate is now left out, and the rest
        of the design is treated exactly as it would be without it, under
        fixed-X (n = 300) and model-X (n = 50) alike."""
        rng = np.random.default_rng(15)
        features, target, _ = _gaussian_design(rng, n=n, n_signal=12, n_noise=13)
        duplicated = np.column_stack([features, features[:, 20]])
        base = KnockoffSelector(random_state=0).select(features, target)
        result = KnockoffSelector(random_state=0).select(duplicated, target)
        assert result.statistics[-1] == 0.0
        np.testing.assert_array_equal(result.statistics[:-1], base.statistics)
        np.testing.assert_array_equal(result.selected, base.selected)
        assert any(
            "linearly dependent" in message and "(columns 25)" in message
            for message in result.warnings_raised
        )

    def test_duplicate_keeps_the_earlier_column(self):
        rng = np.random.default_rng(16)
        features, target, truth = _gaussian_design(rng, n_signal=12, n_noise=13)
        duplicated = np.column_stack([features, features[:, 0]])
        result = KnockoffSelector(construction="fixed", random_state=0).select(duplicated, target)
        assert result.statistics[-1] == 0.0
        assert 0 in result.selected
        assert _fdp(result.selected, np.append(truth, True)) <= 0.2

    def test_rank_deficient_pool_does_not_select_nulls(self):
        """Three signals cannot pass knockoff+ at level 0.1, which needs ten
        positive statistics; with one duplicated null the collapsed
        construction selected 11 to 15 features on three of these four
        seeds, most of them nulls."""
        for seed in range(6210, 6214):
            rng = np.random.default_rng(seed)
            features, target, truth = _gaussian_design(rng, n_signal=3, n_noise=27, effect=1.0)
            duplicated = np.column_stack([features, features[:, 3]])
            result = KnockoffSelector(random_state=seed - 6200).select(duplicated, target)
            assert _fdp(result.selected, np.append(truth, False)) == 0.0

    def test_dependent_columns_count_against_n_for_fixed_x(self):
        """Routing and the fixed-X row requirement use the retained columns."""
        rng = np.random.default_rng(17)
        features = rng.standard_normal((41, 20))
        target = features[:, 0] + rng.standard_normal(41)
        widened = np.column_stack([features, features[:, :3] @ np.ones(3), np.ones(41)])
        result = KnockoffSelector(construction="fixed").select(widened, target)
        np.testing.assert_array_equal(result.statistics[20:], [0.0, 0.0])
        wider = np.column_stack([features, rng.standard_normal(41), features[:, 0]])
        with pytest.raises(ValueError, match="p=21 \\(22 before dropping"):
            KnockoffSelector(construction="fixed").select(wider, target)

    def test_narrow_design_warns_for_knockoff_plus(self):
        rng = np.random.default_rng(12)
        features, target, _ = _gaussian_design(rng, n=200, n_signal=2, n_noise=3)
        result = KnockoffSelector(target_fdr=0.1, offset=1).select(features, target)
        assert any("at least 10 features" in message for message in result.warnings_raised)

    @pytest.mark.parametrize(("target_fdr", "minimum"), [(0.1, 10), (0.3, 4), (0.2, 5)])
    def test_knockoff_plus_floor_is_ceil_of_one_over_q(self, target_fdr, minimum):
        """(1 + #negatives) / #positives <= q needs at least ceil(1/q)
        positives: one feature fewer cannot select, exactly that many can."""
        rng = np.random.default_rng(14)
        for p, satisfiable in ((minimum - 1, False), (minimum, True)):
            features = rng.standard_normal((400, p))
            target = features @ np.full(p, 2.0) + rng.standard_normal(400)
            result = KnockoffSelector(target_fdr=target_fdr).select(features, target)
            warned = any("to be satisfiable" in message for message in result.warnings_raised)
            assert warned is not satisfiable
            assert (result.n_selected == p) is satisfiable

    def test_model_x_noise_independent_of_data_from_the_same_seed(self):
        """Drawn from random_state itself, the model-X noise equalled a data
        matrix generated from the same seed and shape, making the knockoffs a
        fixed linear image of the features."""
        from beamfeat.selection import _standardise

        n, p = 60, 40
        raw = np.random.default_rng(0).standard_normal((n, p))
        standardised = _standardise(raw)
        selector = KnockoffSelector(random_state=0)
        _, smallest, precision = selector._covariance(standardised)
        s_diag = np.diag(np.full(p, min(2.0 * max(smallest, 1e-10), 1.0)))
        mean = standardised - standardised @ precision @ s_diag
        factor = selector._matrix_sqrt(2.0 * s_diag - s_diag @ precision @ s_diag)
        noise = np.linalg.lstsq(factor, (selector._model_x_knockoffs(standardised) - mean).T, rcond=None)[0].T
        assert not np.allclose(noise, raw, atol=1e-3)

    def test_classification_records_heuristic_warning(self):
        rng = np.random.default_rng(13)
        features, target, _ = _gaussian_design(rng, effect=4.0)
        labels = (target > np.median(target)).astype(int)
        result = KnockoffSelector(problem_type="classification").select(features, labels)
        assert any("heuristic" in message for message in result.warnings_raised)

    def test_classification_with_three_classes(self):
        # From scikit-learn 1.8 the liblinear solver fits two classes only, so
        # the statistic is computed one class against the rest; it raised
        # before. The classification path is heuristic, so only power is
        # checked here.
        rng = np.random.default_rng(14)
        features, target, signal = _gaussian_design(rng, effect=4.0)
        labels = np.digitize(target, np.quantile(target, [1 / 3, 2 / 3]))
        result = KnockoffSelector(problem_type="classification", offset=0, target_fdr=0.2).select(
            features, labels
        )
        assert set(np.flatnonzero(signal)) <= set(result.selected.tolist())

    def test_model_x_warning_names_its_reason(self):
        rng = np.random.default_rng(15)
        features, target, _ = _gaussian_design(rng, n=400, n_signal=3, n_noise=7)
        requested = KnockoffSelector(construction="gaussian").select(features, target)
        assert any("construction='gaussian'" in message for message in requested.warnings_raised)
        assert not any("falling back" in message for message in requested.warnings_raised)
        narrow = KnockoffSelector().select(features[:15], target[:15])
        assert any("n <= 2p: falling back" in message for message in narrow.warnings_raised)


# --------------------------------------------------------------------------- #
# Factory and validation
# --------------------------------------------------------------------------- #


class TestFactory:
    @pytest.mark.parametrize("name", ALL_SELECTORS)
    def test_resolves_canonical_names(self, name):
        assert make_selector(name).name == name

    @pytest.mark.parametrize(
        ("alias", "expected"),
        [("knockoffs", "knockoff"), ("model_x", "knockoff"), ("fixed_x", "knockoff"), ("perm", "permutation")],
    )
    def test_resolves_aliases(self, alias, expected):
        assert make_selector(alias).name == expected

    @pytest.mark.parametrize(("alias", "construction"), [("fixed_x", "fixed"), ("model_x", "gaussian"), ("knockoff", "auto")])
    def test_construction_aliases_select_their_construction(self, alias, construction):
        assert make_selector(alias).construction == construction

    def test_explicit_construction_overrides_the_alias(self):
        assert make_selector("fixed_x", construction="auto").construction == "auto"

    def test_instance_passes_through(self):
        selector = PermutationSelector()
        assert make_selector(selector) is selector

    def test_unknown_name_rejected(self):
        with pytest.raises(ValueError, match="unknown selector"):
            make_selector("astrology")

    def test_wrong_type_rejected(self):
        with pytest.raises(TypeError):
            make_selector(3.14)

    def test_kwargs_forwarded(self):
        assert make_selector("permutation", n_permutations=99).n_permutations == 99
        assert make_selector("permutation", correction="by").correction == "by"


class TestValidation:
    @pytest.mark.parametrize("bad_fdr", [0.0, 1.0, -0.1, 1.5])
    def test_bad_fdr_rejected(self, bad_fdr):
        with pytest.raises(ValueError, match="target_fdr"):
            PermutationSelector(target_fdr=bad_fdr)

    def test_bad_problem_type_rejected(self):
        with pytest.raises(ValueError, match="problem_type"):
            PermutationSelector(problem_type="ranking")

    def test_bad_correction_rejected(self):
        with pytest.raises(ValueError, match="correction"):
            PermutationSelector(correction="bonferroni")

    def test_bad_offset_rejected(self):
        with pytest.raises(ValueError, match="offset"):
            KnockoffSelector(offset=5)

    def test_bad_construction_rejected(self):
        with pytest.raises(ValueError, match="construction"):
            KnockoffSelector(construction="quantum")

    def test_too_few_permutations_rejected(self):
        with pytest.raises(ValueError, match="n_permutations"):
            PermutationSelector(n_permutations=1)

    def test_cap_below_floor_rejected(self):
        with pytest.raises(ValueError, match="max_permutations"):
            PermutationSelector(n_permutations=500, max_permutations=100)

    @pytest.mark.parametrize("name", ALL_SELECTORS)
    def test_row_mismatch_rejected(self, name):
        rng = np.random.default_rng(0)
        with pytest.raises(ValueError, match="rows"):
            _fast(name).select(rng.standard_normal((100, 5)), rng.standard_normal(50))

    @pytest.mark.parametrize("name", ALL_SELECTORS)
    def test_non_finite_features_rejected(self, name):
        rng = np.random.default_rng(0)
        features = rng.standard_normal((100, 5))
        features[3, 2] = np.nan
        with pytest.raises(ValueError, match="non-finite"):
            _fast(name).select(features, rng.standard_normal(100))

    @pytest.mark.parametrize("name", ALL_SELECTORS)
    @pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
    def test_non_finite_target_rejected(self, name, bad):
        """A NaN target made every permutation statistic NaN, so no null draw
        counted as an exceedance and every candidate was selected."""
        rng = np.random.default_rng(0)
        target = rng.standard_normal(100)
        target[7] = bad
        with pytest.raises(ValueError, match="target contains"):
            _fast(name).select(rng.standard_normal((100, 5)), target)

    def test_missing_class_label_rejected(self):
        rng = np.random.default_rng(0)
        labels = np.array(["a", "b"] * 50, dtype=object)
        labels[4] = None
        with pytest.raises(ValueError, match="target contains"):
            PermutationSelector(problem_type="classification").select(rng.standard_normal((100, 3)), labels)

    def test_string_class_labels_still_accepted(self):
        rng = np.random.default_rng(0)
        labels = np.array(["a", "b"] * 50, dtype=object)
        result = PermutationSelector(problem_type="classification").select(rng.standard_normal((100, 3)), labels)
        assert result.n_candidates == 3

    @pytest.mark.parametrize("name", ALL_SELECTORS)
    def test_constant_column_handled(self, name):
        rng = np.random.default_rng(0)
        features = np.column_stack([np.ones(200), rng.standard_normal(200)])
        result = _fast(name).select(features, rng.standard_normal(200))
        assert np.all(np.isfinite(result.statistics))

    def test_constant_target_handled(self):
        rng = np.random.default_rng(0)
        result = PermutationSelector().select(rng.standard_normal((100, 4)), np.ones(100))
        assert result.n_selected == 0


class TestAdjustedPValues:
    def test_q_values_agree_with_selection_decision(self):
        """A feature is selected iff its q-value is at or below target_fdr —
        the defining property of a step-up adjusted p-value."""
        rng = np.random.default_rng(0)
        features, target, _ = _gaussian_design(rng)
        for correction in ("bh", "by"):
            result = PermutationSelector(
                target_fdr=0.2, random_state=0, correction=correction
            ).select(features, target)
            by_q = np.flatnonzero(result.q_values <= result.target_fdr + 1e-12)
            np.testing.assert_array_equal(np.sort(by_q), result.selected)

    def test_q_values_dominate_p_values_and_by_dominates_bh(self):
        from beamfeat.selection import _adjusted_p_values

        rng = np.random.default_rng(1)
        p_values = rng.uniform(0, 1, 30)
        q_bh = _adjusted_p_values(p_values, "bh")
        q_by = _adjusted_p_values(p_values, "by")
        assert np.all(q_bh >= p_values - 1e-12)
        assert np.all(q_by >= q_bh - 1e-12)
        assert np.all(q_by <= 1.0)


class TestKnockoffUnsatisfiabilityDiagnostic:
    def test_empty_knockoff_plus_reports_offset_zero_alternative(self):
        """When knockoff+ selects nothing but offset=0 would have selected,
        the result must say so — an empty result should read as "could not
        have selected at this configuration", not "no evidence"."""
        rng = np.random.default_rng(0)
        features, target, _ = _gaussian_design(rng, n=300, n_signal=2, n_noise=4)
        result = KnockoffSelector(target_fdr=0.1, offset=1, random_state=0).select(features, target)
        if result.n_selected == 0:
            assert any("offset=0" in message for message in result.warnings_raised)


class TestPermutationResolution:
    """The permutation budget must be large enough for the *configured*
    correction, not merely for Benjamini-Hochberg.

    The smallest p-value the add-one estimator can return is 1/(B+1). BH
    asks the leading feature to clear q/m; BY asks it to clear q/(m c(m)),
    a threshold smaller by the harmonic factor. Sizing B to the BH bound
    while correcting with BY leaves the procedure unable to reject anything
    at all, and it fails silently: the estimator reports an empty selection,
    which reads as "no signal" rather than "budget too small".
    """

    @staticmethod
    def _single_signal(rng, n=400, m=400, effect=3.0):
        """One overwhelmingly strong column among m nulls.

        The single-signal case is the one that binds. Several tied true
        signals relax the threshold by their count, which is why the failure
        does not show up on designs with a handful of planted columns.
        """
        features = rng.standard_normal((n, m))
        target = rng.standard_normal(n)
        features[:, 0] += effect * target
        return features, target

    def test_by_bound_scales_by_the_harmonic_factor(self):
        for m in (25, 400, 1200):
            harmonic = float(np.sum(1.0 / np.arange(1, m + 1)))
            bh = PermutationSelector(target_fdr=0.1, correction="bh")._required_permutations(m)
            by = PermutationSelector(target_fdr=0.1, correction="by")._required_permutations(m)
            assert by == pytest.approx(bh * harmonic, rel=1e-3)

    @pytest.mark.parametrize("m", [100, 400, 1200])
    def test_by_selects_a_lone_strong_signal(self, m):
        rng = np.random.default_rng(0)
        features, target = self._single_signal(rng, m=m)
        result = PermutationSelector(target_fdr=0.1, correction="by", random_state=0).select(
            features, target
        )
        assert result.n_selected >= 1
        assert 0 in result.selected

    def test_bh_budget_is_unchanged(self):
        """The fix must not inflate the Benjamini-Hochberg path, whose bound
        was already correct."""
        selector = PermutationSelector(target_fdr=0.1, correction="bh")
        assert selector._required_permutations(500) == int(np.ceil(2 * 500 / 0.1))

    def test_fixed_budget_below_the_bound_warns(self):
        """With auto_permutations off, a budget whose floor 1/(B+1) cannot
        clear the correction's threshold used to return an empty selection
        in silence."""
        rng = np.random.default_rng(0)
        features, target = self._single_signal(rng, n=300, m=100)
        result = PermutationSelector(n_permutations=200, auto_permutations=False).select(features, target)
        assert result.n_selected == 0
        assert any("auto_permutations=False" in message for message in result.warnings_raised)

    def test_fixed_budget_above_the_bound_is_silent(self):
        rng = np.random.default_rng(0)
        features, target = self._single_signal(rng, n=300, m=100)
        result = PermutationSelector(n_permutations=3000, auto_permutations=False).select(features, target)
        assert not result.warnings_raised
        assert 0 in result.selected

    def test_unsatisfiable_budget_names_the_correction(self):
        rng = np.random.default_rng(0)
        features, target = self._single_signal(rng, n=60, m=300)
        result = PermutationSelector(
            target_fdr=0.1,
            correction="by",
            random_state=0,
            n_permutations=100,
            max_permutations=100,
        ).select(features, target)
        assert any("BY" in message for message in result.warnings_raised)


class TestConditionalPermutation:
    """The Freedman-Lane test behind ``certify_test="conditional"``: each
    feature is tested given the others, so a term that only rides on a true
    one fails where the marginal test passes it."""

    @staticmethod
    def _rider_design(rng, n=600):
        X = rng.uniform(0.5, 3.0, (n, 8))
        y = 2 * X[:, 0] + rng.normal(0, 0.5, n)
        return np.column_stack([X[:, 0], X[:, 0] * X[:, 7]]), y

    def test_drops_a_rider_the_marginal_test_keeps(self):
        features, target = self._rider_design(np.random.default_rng(1))
        marginal = PermutationSelector(correction="by", random_state=1).select(features, target)
        conditional = _ConditionalPermutationSelector(correction="by", random_state=1).select(
            features, target
        )
        assert marginal.selected.tolist() == [0, 1]
        assert conditional.selected.tolist() == [0]
        assert conditional.method == "conditional_permutation"
        assert conditional.p_values[1] > 0.5

    def test_keeps_every_true_term(self):
        rng = np.random.default_rng(2)
        X = rng.uniform(0.5, 3.0, (300, 4))
        y = X[:, 0] + X[:, 1] * X[:, 2] + rng.normal(0, 0.5, 300)
        features = np.column_stack([X[:, 0], X[:, 1] * X[:, 2], X[:, 3]])
        result = _ConditionalPermutationSelector(correction="by", random_state=0).select(features, y)
        assert result.selected.tolist() == [0, 1]

    def test_null_rejection_rate_is_at_most_q(self):
        """A term that adds nothing given the others is rejected at a rate of
        at most the level, before and after the step-up correction."""
        q, draws = 0.1, 200
        raw, fdps = 0, []
        for draw in range(draws):
            rng = np.random.default_rng(5000 + draw)
            X = rng.uniform(0.5, 3.0, (80, 3))
            y = 2 * X[:, 0] + rng.normal(0, 1.0, 80)
            # x0 is true; x0*x1 rides on it, x2 is noise: both are null given the rest.
            features = np.column_stack([X[:, 0], X[:, 0] * X[:, 1], X[:, 2]])
            result = _ConditionalPermutationSelector(
                target_fdr=q, correction="bh", random_state=draw, n_permutations=500
            ).select(features, y)
            raw += int(result.p_values[1] <= q)
            fdps.append(float(np.isin(result.selected, [1, 2]).sum()) / max(result.n_selected, 1))
        # 0.1 plus three standard errors of a proportion over 200 draws.
        assert raw / draws <= q + 3 * np.sqrt(q * (1 - q) / draws)
        assert float(np.mean(fdps)) <= q

    def test_a_single_feature_is_the_marginal_test(self):
        rng = np.random.default_rng(3)
        x = rng.normal(size=(40, 1))
        y = 0.5 * x[:, 0] + rng.normal(size=40)
        conditional = _ConditionalPermutationSelector(random_state=0, n_permutations=20000).select(x, y)
        marginal = PermutationSelector(random_state=0, n_permutations=20000).select(x, y)
        assert conditional.p_values[0] == pytest.approx(marginal.p_values[0], abs=0.01)

    def test_a_linear_combination_of_the_others_cannot_be_tested(self):
        rng = np.random.default_rng(4)
        X = rng.normal(size=(50, 3))
        features = np.column_stack([X, X[:, 0] - 2 * X[:, 1]])
        result = _ConditionalPermutationSelector(random_state=0).select(features, X[:, 2] + rng.normal(size=50))
        assert result.p_values[[0, 1, 3]].tolist() == [1.0, 1.0, 1.0]
        assert 2 in result.selected
        assert any("could not be tested" in message for message in result.warnings_raised)

    def test_too_few_rows_for_the_features_cannot_be_tested(self):
        rng = np.random.default_rng(5)
        features = rng.normal(size=(6, 5))
        result = _ConditionalPermutationSelector(random_state=0).select(features, rng.normal(size=6))
        assert result.n_selected == 0
        assert np.all(result.p_values == 1.0)
        assert any("could not be tested" in message for message in result.warnings_raised)

    def test_classification_is_refused(self):
        rng = np.random.default_rng(6)
        features = rng.normal(size=(40, 2))
        with pytest.raises(ValueError, match="regression"):
            _ConditionalPermutationSelector(problem_type="classification").select(
                features, (features[:, 0] > 0).astype(int)
            )

    def test_budget_follows_the_correction(self):
        selector = _ConditionalPermutationSelector(target_fdr=0.1, correction="by")
        assert selector._required_permutations(40) == PermutationSelector(
            target_fdr=0.1, correction="by"
        )._required_permutations(40)


# --------------------------------------------------------------------------- #
# Fixed-X knockoff statistic
# --------------------------------------------------------------------------- #


class TestLassoEntryStatistic:
    """The fixed-X path's statistic sees the data only through X'X and X'y."""

    @staticmethod
    def _centred(rng, n=120, p=6):
        X = rng.standard_normal((n, p))
        X -= X.mean(axis=0)
        return X, X[:, 0] * 2.0 + rng.standard_normal(n)

    def test_unchanged_by_a_rotation_of_the_rows(self):
        # An orthogonal Q with Q1 = 1 keeps X'X, X'y and the centring, while
        # moving every row; a cross-validated statistic would change.
        rng = np.random.default_rng(0)
        X, y = self._centred(rng)
        v = rng.standard_normal(X.shape[0])
        v -= v.mean()
        v /= np.linalg.norm(v)
        Q = np.eye(X.shape[0]) - 2.0 * np.outer(v, v)
        np.testing.assert_allclose(
            Selector._lasso_entry(Q @ X, Q @ y), Selector._lasso_entry(X, y), rtol=1e-9, atol=1e-12
        )

    def test_the_signal_enters_first(self):
        X, y = self._centred(np.random.default_rng(1))
        entry = Selector._lasso_entry(X, y)
        assert int(np.argmax(entry)) == 0
        assert np.all(entry >= 0.0)

    def test_fixed_x_selects_the_signals(self):
        features, target, truth = _gaussian_design(np.random.default_rng(3), n=400, n_noise=35)
        result = KnockoffSelector(target_fdr=0.2, construction="fixed", random_state=0).select(
            features, target
        )
        assert set(np.flatnonzero(truth)) <= set(result.selected.tolist())


# --------------------------------------------------------------------------- #
# Input test
# --------------------------------------------------------------------------- #


class TestInputPermutationTest:
    """Each column inside each candidate, tested by permuting it alone."""

    @staticmethod
    def _data(seed=0, n=300):
        rng = np.random.default_rng(seed)
        X = rng.uniform(1.0, 6.0, (n, 4))
        data = {f"x{j}": X[:, j] for j in range(4)}
        return data, X[:, 0] * X[:, 1] + rng.normal(0, 1.0, n)

    @staticmethod
    def _nodes():
        x0, x1, x2 = leaf("x0"), leaf("x1"), leaf("x2")
        product = combine("mul", x0, x1)
        rider = combine("div", product, combine("add", x1, x2))
        return [product, rider, x0, transform("log", x0)]

    def test_keeps_the_signal_columns_and_flags_the_rider(self):
        data, y = self._data()
        pairs, result = _InputPermutationTest(target_fdr=0.1, correction="by").test_inputs(
            self._nodes(), data, y
        )
        passed = {pairs[i] for i in result.selected}
        assert pairs == [(0, "x0"), (0, "x1"), (1, "x0"), (1, "x1"), (1, "x2")]
        assert {(0, "x0"), (0, "x1")} <= passed
        assert (1, "x2") not in passed
        assert result.n_candidates == 5 and result.method == "input_permutation"

    def test_single_column_candidates_are_not_tested(self):
        data, y = self._data()
        pairs, result = _InputPermutationTest().test_inputs([leaf("x0"), transform("log", leaf("x1"))], data, y)
        assert pairs == [] and result.n_candidates == 0

    def test_null_column_p_values_are_valid(self):
        # x2 is independent of the target and of x0, so its p-value inside
        # x0*x2 is super-uniform: at most about 10% fall at or below 0.1.
        x0, x2 = leaf("x0"), leaf("x2")
        node = combine("mul", x0, x2)
        tester = _InputPermutationTest(n_permutations=99, auto_permutations=False, correction="bh")
        small = 0
        for seed in range(200):
            rng = np.random.default_rng(seed)
            X = rng.uniform(1.0, 6.0, (60, 3))
            data = {f"x{j}": X[:, j] for j in range(3)}
            tester.random_state = seed
            pairs, result = tester.test_inputs([node], data, X[:, 0] + rng.normal(0, 1.0, 60))
            small += result.p_values[pairs.index((0, "x2"))] <= 0.1
        assert small / 200 <= 0.1 + 3 * np.sqrt(0.1 * 0.9 / 200)

    def test_stops_early_on_a_plainly_unneeded_column(self):
        data, y = self._data()
        pairs, result = _InputPermutationTest(target_fdr=0.1).test_inputs(self._nodes(), data, y)
        p_rider = result.p_values[pairs.index((1, "x2"))]
        # Stopped at its tenth exceedance: 10/L for some L well short of B.
        assert p_rider > 0.01 and np.isclose(10.0 / p_rider, round(10.0 / p_rider))

    def test_non_finite_permutations_count_against_the_column(self):
        rng = np.random.default_rng(0)
        x = rng.uniform(1.0, 2.0, 80)
        shift = np.concatenate([-x[:40] + 0.5, rng.uniform(0.0, 1.0, 40)])
        data = {"x0": x, "x1": shift}
        node = transform("log", combine("add", leaf("x0"), leaf("x1")))
        target = np.log(x + shift) + rng.normal(0, 0.01, 80)
        pairs, result = _InputPermutationTest(n_permutations=99, auto_permutations=False).test_inputs(
            [node], data, target
        )
        # Permuting x1 puts log's operand below zero on almost every draw.
        assert result.p_values[pairs.index((0, "x1"))] > 0.05

    @pytest.mark.parametrize("n_jobs", [2, -1])
    def test_same_p_values_for_every_n_jobs(self, n_jobs):
        data, y = self._data()
        serial = _InputPermutationTest(target_fdr=0.1).test_inputs(self._nodes(), data, y)
        threaded = _InputPermutationTest(target_fdr=0.1, n_jobs=n_jobs).test_inputs(self._nodes(), data, y)
        assert serial[0] == threaded[0]
        np.testing.assert_array_equal(serial[1].p_values, threaded[1].p_values)

    def test_classification(self):
        data, y = self._data()
        labels = (y > np.median(y)).astype(int)
        pairs, result = _InputPermutationTest(problem_type="classification").test_inputs(
            self._nodes()[:1], data, labels
        )
        assert set(result.selected.tolist()) == {0, 1}

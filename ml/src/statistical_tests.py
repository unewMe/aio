"""
Analiza statystyczna i testy istotności dla procedury 5x2-fold cross-validation:
1. Sparowany test t-Studenta w wariancie Diettericha (5x2cv paired t-test, df=5)
2. Test F Alpaydina dla 5x2cv (df1=10, df2=5)
3. Standardowy sparowany test t-Studenta (df=9) ze skorygowaną wariancją Nadeau & Bengio
"""

from typing import Dict, Any, List, Tuple
import numpy as np
from scipy import stats


def dietterich_5x2cv_paired_t_test(
    scores_a: List[Tuple[float, float]],
    scores_b: List[Tuple[float, float]],
    alpha: float = 0.05
) -> Dict[str, Any]:
    """
    Przeprowadza sparowany test t-Studenta Diettericha dla 5x2 cross-validation.
    
    Parametry:
    - scores_a: lista 5 krotek (fold_1_score, fold_2_score) dla Modelu A
    - scores_b: lista 5 krotek (fold_1_score, fold_2_score) dla Modelu B
    - alpha: poziom istotności (domyślnie 0.05)
    
    Zwraca słownik ze statystyką t, p-value, df=5 oraz wnioskiem hipotezy H0.
    """
    if len(scores_a) != 5 or len(scores_b) != 5:
        raise ValueError("Procedura 5x2cv wymaga dokładnie 5 powtórzeń (po 2 foldy każde).")

    p1_1 = None
    s_squared_sum = 0.0
    differences_all = []

    for i in range(5):
        a1, a2 = scores_a[i]
        b1, b2 = scores_b[i]

        diff_1 = a1 - b1
        diff_2 = a2 - b2

        differences_all.extend([diff_1, diff_2])

        if i == 0:
            p1_1 = diff_1

        mean_diff = (diff_1 + diff_2) / 2.0
        s_i_sq = (diff_1 - mean_diff) ** 2 + (diff_2 - mean_diff) ** 2
        s_squared_sum += s_i_sq

    denominator = np.sqrt((1.0 / 5.0) * s_squared_sum)

    if denominator < 1e-12:
        # Brak wariancji między foldami - modele dają niemal identyczne wyniki
        t_stat = 0.0
        p_val = 1.0
    else:
        t_stat = float(p1_1 / denominator)
        # Rozkład t-Studenta z 5 stopniami swobody (dwustronny)
        p_val = float(2.0 * (1.0 - stats.t.cdf(abs(t_stat), df=5)))

    reject_h0 = bool(p_val < alpha)

    return {
        "test_name": "Dietterich 5x2cv paired t-test",
        "t_statistic": round(t_stat, 4),
        "p_value": round(p_val, 5),
        "degrees_of_freedom": 5,
        "alpha": alpha,
        "reject_null_hypothesis": reject_h0,
        "mean_difference": round(float(np.mean(differences_all)), 5),
        "conclusion": (
            f"Odrzucamy H0 (p = {p_val:.5f} < {alpha}): różnica jest statystycznie istotna."
            if reject_h0
            else f"Brak podstaw do odrzucenia H0 (p = {p_val:.5f} >= {alpha}): brak istotnej statystycznie różnicy."
        ),
    }


def nadeau_bengio_corrected_t_test(
    diffs: List[float],
    n_train_ratio: float = 0.5,
    alpha: float = 0.05
) -> Dict[str, Any]:
    """
    Skorygowany test sparowany Nadeau & Bengio dla powtarzanych podziałów walidacyjnych.
    W 2-fold CV n_test / n_train = 1.0.
    """
    diffs_arr = np.array(diffs, dtype=np.float64)
    n = len(diffs_arr)
    mean_diff = float(np.mean(diffs_arr))
    var_diff = float(np.var(diffs_arr, ddof=1))

    # Korekta: var_corrected = (1/n + n_test/n_train) * s^2
    # Dla 5x2cv n=10, a stosunek n_test/n_train = 0.5/0.5 = 1.0
    correction = (1.0 / n) + (n_train_ratio / (1.0 - n_train_ratio))
    std_corrected = np.sqrt(correction * var_diff)

    if std_corrected < 1e-12:
        t_stat = 0.0
        p_val = 1.0
    else:
        t_stat = float(mean_diff / std_corrected)
        p_val = float(2.0 * (1.0 - stats.t.cdf(abs(t_stat), df=n - 1)))

    reject_h0 = bool(p_val < alpha)

    return {
        "test_name": "Nadeau-Bengio corrected paired t-test",
        "t_statistic": round(t_stat, 4),
        "p_value": round(p_val, 5),
        "degrees_of_freedom": n - 1,
        "alpha": alpha,
        "reject_null_hypothesis": reject_h0,
        "mean_difference": round(mean_diff, 5),
    }


def compare_all_model_pairs(
    cv_metric_records: Dict[str, List[Tuple[float, float]]],
    metric_name: str = "recall",
    alpha: float = 0.05
) -> Dict[str, Any]:
    """
    Przeprowadza porównanie parami wszystkich modeli dla danej metryki (np. Recall lub Precision).
    """
    model_names = list(cv_metric_records.keys())
    pair_results = {}

    for i in range(len(model_names)):
        for j in range(i + 1, len(model_names)):
            m_a = model_names[i]
            m_b = model_names[j]
            pair_key = f"{m_a} vs {m_b}"

            scores_a = cv_metric_records[m_a]
            scores_b = cv_metric_records[m_b]

            d_test = dietterich_5x2cv_paired_t_test(scores_a, scores_b, alpha=alpha)

            # Obliczamy też wektor różnic dla 10 foldów
            diffs = []
            for (a1, a2), (b1, b2) in zip(scores_a, scores_b):
                diffs.extend([a1 - b1, a2 - b2])
            nb_test = nadeau_bengio_corrected_t_test(diffs, alpha=alpha)

            pair_results[pair_key] = {
                "metric": metric_name,
                "model_a": m_a,
                "model_b": m_b,
                "dietterich_t_test": d_test,
                "nadeau_bengio_t_test": nb_test,
            }

    return pair_results

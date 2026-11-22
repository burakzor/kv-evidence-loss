from __future__ import annotations

def select_paragraphs(ranked_titles: list[str], paragraph_lengths: dict[str, int], n_other_tokens: int, n_tokens: int, budget: float) -> list[str]:
    """Return the top k ranked titles whose kept tokens (the k paragraphs plus n_other_tokens) come closest to budget * n_tokens, with k at least 1 and ties going to the smaller k."""
    target = budget * n_tokens
    best_k = 1
    best_diff = None
    n_kept = n_other_tokens
    for k, title in enumerate(ranked_titles, start=1):
        n_kept += paragraph_lengths[title]
        diff = abs(n_kept - target)
        if best_diff is None or diff < best_diff:
            best_k = k
            best_diff = diff
    return ranked_titles[:best_k]

import numpy as np
from scipy.special import comb

def unbiased_max_at_k(g_sorted, k):
    """Expected maximum when drawing k items uniformly without replacement."""
    if k == 0: return 0.0
    n = len(g_sorted)
    if n < k: return 0.0
    i_idx = np.arange(k - 1, n)
    return np.sum(comb(i_idx, k - 1) * g_sorted[i_idx]) / comb(n, k)

def conditional_unbiased_max_at_k(g_array, k):
    """E[max(S) | S contains item i] over uniform random k-subset S, for each i (Eq.19 s)."""
    n = len(g_array)
    sort_idx = np.argsort(g_array)
    g_sorted = g_array[sort_idx]
    
    s_sorted = np.zeros(n)
    # Average over k-subsets that contain each item: there are C(n-1, k-1) of them.
    denom = comb(n - 1, k - 1)
    
    # Diagonals (when the item itself is the maximum of the k subset)
    i_diag = np.arange(k - 1, n)
    s_sorted[i_diag] += comb(i_diag, k - 1) * g_sorted[i_diag]
    
    # Off-diagonals (when a larger item j is the maximum)
    j_idx = np.arange(k - 1, n)
    terms = comb(j_idx - 1, k - 2) * g_sorted[j_idx] if k >= 2 else np.zeros_like(j_idx)
    
    all_terms = np.zeros(n)
    all_terms[j_idx] = terms
    suffix_sums = np.cumsum(all_terms[::-1])[::-1]
    
    i_off = np.arange(n - 1)
    s_sorted[i_off] += suffix_sums[i_off + 1]
    
    s_sorted /= denom
    
    # Restore original ordering
    s_out = np.zeros(n)
    s_out[sort_idx] = s_sorted
    return s_out

def get_s_loo_vec(g_array, k):
    """Computes 's_loo' (Eq 29) using Leave-One-Out subset baseline."""
    n = len(g_array)
    s_unb = conditional_unbiased_max_at_k(g_array, k)
    s_out = np.zeros(n)
    for i in range(n):
        g_minus = np.delete(g_array, i)
        # Expected max of k draws from the (N-1) items excluding item i
        s_out[i] = s_unb[i] - unbiased_max_at_k(np.sort(g_minus), k)
    return s_out

def get_s_loo_minus_one_vec(g_array, k):
    """Computes 's_loo_minus_one' (Eq 33) baseline using (k-1) subsets."""
    n = len(g_array)
    s_unb = conditional_unbiased_max_at_k(g_array, k)
    s_out = np.zeros(n)
    for i in range(n):
        g_minus = np.delete(g_array, i)
        g_sorted = np.sort(g_minus)
        s_out[i] = s_unb[i] - unbiased_max_at_k(g_sorted, k-1)
    return s_out

if __name__ == "__main__":
    g_array = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    k = 3
    print(unbiased_max_at_k(g_array, k))
    # simulate:
    # iterate through all k-subsets of g_array
    from itertools import combinations
    res = 0
    for subset in combinations(g_array, k):
        print(subset, np.max(subset))
        res += np.max(subset)
    print(res / 10)

    print(conditional_unbiased_max_at_k(g_array, k))
    # simulate:
    for leave_idx in range(len(g_array)):
        g_minus = np.delete(g_array, leave_idx)
        res = 0
        for subset in combinations(g_minus, k-1):
            res += np.max(np.concatenate([subset, [g_array[leave_idx]]]))
        print(res / len(list(combinations(g_minus, k-1))))
            
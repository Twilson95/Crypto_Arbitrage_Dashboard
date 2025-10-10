import os
import numpy as np
import pandas as pd
from glob import glob
from statsmodels.tsa.vector_ar.vecm import coint_johansen
from sklearn.decomposition import PCA
from statsmodels.tsa.stattools import coint
from matplotlib import pyplot as plt


def load_log_prices(data_folder, min_rows=500):
    files = sorted(glob(os.path.join(data_folder, "*.csv")))
    price_data = {}

    for file in files:
        df = pd.read_csv(file)
        df["datetime"] = pd.to_datetime(df["datetime"])
        df.set_index("datetime", inplace=True)

        if df["close"].count() >= min_rows:
            symbol = os.path.basename(file).replace(".csv", "")
            price_data[symbol] = np.log(df["close"])
            print(f"Including {symbol}: {len(df)} rows")
        else:
            print(f"Skipping {file}: only {len(df)} rows")

    # Combine into a DataFrame
    log_prices = pd.concat(price_data.values(), axis=1, join="inner")
    log_prices.columns = list(price_data.keys())

    print(f"\nFinal shape: {log_prices.shape} (dates x assets)")
    print(f"Date range: {log_prices.index.min()} to {log_prices.index.max()}")
    return log_prices


# def compute_matrices(S, max_lag=5):
#     # M0 = np.cov(S.T)
#     M0 = np.cov(S.T) + 1e-4 * np.eye(S.shape[1])
#
#     Mi = [
#         np.cov(S[max_lag:].T, S.shift(-i)[max_lag:].T)[0 : S.shape[1], S.shape[1] :]
#         for i in range(1, max_lag + 1)
#     ]
#     return M0, Mi


def compute_matrices(S, max_lag=5):
    N = S.shape[1]
    M0 = np.cov(S.T) + 1e-4 * np.eye(N)  # regularize

    Mi = []
    for i in range(1, max_lag + 1):
        S1 = S[max_lag:]
        S2 = S.shift(-i)[max_lag:]
        combined = pd.concat([S1, S2], axis=1).dropna()

        if combined.shape[0] < 2:
            Mi.append(np.full((N, N), np.nan))
            continue

        try:
            cov = np.cov(combined.T)
            if cov.shape[0] == 2 * N:
                Mi.append(cov[:N, N:])
            else:
                Mi.append(np.full((N, N), np.nan))
        except:
            Mi.append(np.full((N, N), np.nan))

    return M0, Mi


def projection_onto_l1_ball(v, B):
    if B <= 0:
        return np.zeros_like(v)

    u = np.abs(v)
    if u.sum() <= B:
        return v

    sorted_u = np.sort(u)[::-1]
    cssv = np.cumsum(sorted_u)
    rho_candidates = sorted_u - (cssv - B) / (np.arange(len(u)) + 1)
    rho_indices = np.nonzero(rho_candidates > 0)[0]

    if len(rho_indices) == 0:
        return np.zeros_like(v)  # fallback

    rho = rho_indices[-1]
    theta = (cssv[rho] - B) / (rho + 1)
    return np.sign(v) * np.maximum(u - theta, 0)


def solve_admm(A, b, B_mat, B_bound, rho=1.0, max_iter=100):
    N = A.shape[0]
    w = np.zeros(N)
    z = np.zeros(B_mat.shape[0])
    u = np.zeros(B_mat.shape[0])

    BTB = B_mat.T @ B_mat
    inv_term = np.linalg.inv(2 * A + rho * BTB)

    for _ in range(max_iter):
        w = -inv_term @ (b + rho * B_mat.T @ (u - z))
        h = B_mat @ w + u
        z = projection_onto_l1_ball(h, B_bound)
        u = u + B_mat @ w - z

    return w


def compute_loss(w, M0, Mi, loss_function):
    if loss_function == "variance":
        loss = w.T @ M0 @ w

    elif loss_function == "autocovariance":
        loss = 0.0
        for i in range(len(Mi)):
            mat = Mi[i].T @ Mi[i]
            vec = mat @ w
            term = np.linalg.norm(vec) ** 2
            if np.isfinite(term):
                loss += term

    elif loss_function == "portmanteau":
        loss = 0.0
        for i in range(len(Mi)):
            M_sym = Mi[i] + Mi[i].T
            vec = M_sym @ w
            term = np.linalg.norm(vec) ** 2
            if np.isfinite(term):
                loss += term
    else:
        raise ValueError("Unknown loss function")
    return np.nan_to_num(loss, nan=np.inf, posinf=np.inf, neginf=np.inf)


def sca_mrp(
    S,
    B_mat,
    B_bound,
    mu=0.1,
    tau=1e-2,
    max_iter=1000,
    l2_penalty=0.01,
    loss_function="portmanteau",
    patience=10,
    tol=1e-6,
):
    N = S.shape[1]
    M0, Mi = compute_matrices(S)
    w = np.random.randn(N)
    best_loss = np.inf
    no_improve = 0

    for i in range(max_iter):
        if loss_function == "variance":
            A_U = M0
            b_U = -2 * M0 @ w
        elif loss_function == "autocovariance":
            A_U = sum([(Mi[i].T @ Mi[i]) for i in range(len(Mi))])
            b_U = -2 * A_U @ w
        elif loss_function == "portmanteau":
            A_U = np.zeros((N, N))
            for i in range(len(Mi)):
                M_sym = Mi[i] + Mi[i].T
                norm = np.linalg.norm(M_sym)
                if not np.all(np.isfinite(M_sym)) or norm < 1e-8:
                    continue  # skip unstable or tiny matrices
                A_U += (M_sym @ M_sym.T) / (norm + 1e-8)
            b_U = -2 * A_U @ w
        else:
            raise ValueError(
                "Unknown loss function: choose from 'portmanteau', 'variance', 'autocovariance'"
            )

        denom = w.T @ M0 @ w
        if denom < 1e-8:
            return np.zeros_like(w), np.inf  # Bail early

        b_V = -2 * M0 @ w / denom**2
        A_k = A_U + tau * np.eye(N) + l2_penalty * np.eye(N)
        b_k = b_U + mu * b_V - 2 * tau * w
        w_new = solve_admm(A_k, b_k, B_mat, B_bound)
        w = 0.5 * w + 0.5 * w_new
        current_loss = compute_loss(w, M0, Mi, loss_function)

        if current_loss + tol < best_loss:
            best_loss = current_loss
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= patience:
            print(f"early stop at iteration {i}")
            break

    loss = compute_loss(w, M0, Mi, loss_function)
    return w, loss


def run_johansen_test(S, det_order=0, k_ar_diff=1, rank=None):
    result = coint_johansen(S, det_order, k_ar_diff)
    if rank is None:
        # Use trace statistic to determine rank
        trace_stats = result.lr1
        critical_values = result.cvt[:, 1]  # 5% significance
        rank = sum(trace_stats > critical_values)
    B = result.evec[:, :rank]  # N eigenvectors for N cointegrating relationships
    return B


def find_cointegrated_pairs(S, threshold=0.05):
    n = S.shape[1]
    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            score, pvalue, _ = coint(S.iloc[:, i], S.iloc[:, j])
            if pvalue < threshold:
                pairs.append((S.columns[i], S.columns[j], pvalue))
    return pairs


def reduce_assets(S_log, n_components=10):
    pca = PCA(n_components=n_components)
    S_reduced = pd.DataFrame(pca.fit_transform(S_log), index=S_log.index)
    return S_reduced


def select_cointegrated_pairs(S, top_k=16, threshold=0.05):
    pairs = []
    used_assets = set()

    for i in range(S.shape[1]):
        for j in range(i + 1, S.shape[1]):
            a, b = S.columns[i], S.columns[j]
            score, pvalue, _ = coint(S[a], S[b])
            if pvalue < threshold:
                pairs.append((a, b, pvalue))

    # Sort pairs by p-value
    pairs.sort(key=lambda x: x[2])

    selected_assets = []
    selected_pairs = []

    for a, b, p in pairs:
        if a not in used_assets and b not in used_assets:
            selected_pairs.append((a, b))
            selected_assets.extend([a, b])
            used_assets.update([a, b])
        if len(selected_assets) >= top_k * 2:
            break

    return selected_assets, selected_pairs


def build_spread_to_asset_matrix(pairs, asset_list):
    """
    Construct the mapping from spreads to asset weights.
    Each spread is (a - b) → +1 for a, -1 for b
    """
    A = np.zeros((len(asset_list), len(pairs)))
    asset_idx = {asset: i for i, asset in enumerate(asset_list)}

    for j, (a, b) in enumerate(pairs):
        if a in asset_idx:
            A[asset_idx[a], j] = 1
        if b in asset_idx:
            A[asset_idx[b], j] = -1
    return A


data_folder = (
    # "C:/Users/thoma/PycharmProjects/CryptoDashboard/data/historical_data/Kraken/"
    "C:/Users/thoma/PycharmProjects/CryptoDashboard/data/historical_data/Kraken_long_history"
)

# loss_function = "variance"
loss_function = "portmanteau"
# loss_function = "autocovariance"

S_log = load_log_prices(data_folder, min_rows=400)

# Select best cointegrated, non-overlapping asset pairs
selected_assets, selected_pairs = select_cointegrated_pairs(S_log, top_k=16)

# Use only selected assets for MRP design
S_selected = S_log[selected_assets]

# Optional: project to spreads (difference of each pair)
spreads = []
for a, b in selected_pairs:
    spreads.append(S_selected[a] - S_selected[b])
S_spread = pd.concat(spreads, axis=1)
S_spread.columns = [f"{a}-{b}" for a, b in selected_pairs]
S_spread = (S_spread - S_spread.mean()) / S_spread.std()

# Identity matrix since S_spread is already mean-reverting basis
B_mat = np.eye(S_spread.shape[1])

# Run MRP optimization
# w_opt = sca_mrp(S_spread, B_mat, B_bound=1.0, mu=0.1, loss_function=loss_function)
#
# w_avg = np.zeros_like(w_opt)
# for i in range(100):
#     print(f"sca_mrp {i+1}")
#     w = sca_mrp(S_spread, B_mat, B_bound=1.0, mu=0.1, loss_function=loss_function)
#     w_avg += w
# w_avg /= 100

best_loss = np.inf
w_best = None
for i in range(1000):
    print(f"sca_mrp {i + 1}")
    # w, loss = sca_mrp(S_spread, B_mat, B_bound=1.0, mu=0.1, loss_function=loss_function)
    w, loss = sca_mrp(
        S_spread,
        B_mat,
        B_bound=5.0,
        mu=0.5,
        loss_function=loss_function,
        l2_penalty=0.1,
    )

    if loss < best_loss:
        best_loss = loss
        w_best = w
    print(f"loss {loss}, best loss {best_loss}")

A = build_spread_to_asset_matrix(selected_pairs, selected_assets)
w_asset = A @ w_best  # final weights on each asset

for asset, weight in zip(selected_assets, w_asset):
    action = "LONG" if weight > 0 else "SHORT" if weight < 0 else "FLAT"
    print(f"{asset:10s} → {action:5s} | Weight: {weight:.4f}")

# Final mean-reverting spread (portfolio)
spread_final = S_spread @ w_best
spread_final.plot(title="Final MRP Spread from Top 12 Cointegrated Pairs")
plt.show()

"""
Streamlit app: Interactive PCA on MNIST via SVD.

Run with:
    streamlit run app_pca_mnist.py

Project 5 -- Low-Rank Approximation
Numerical Linear Algebra, MSc MMCS, University of Luxembourg
"""

import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from numpy.linalg import svd


st.set_page_config(page_title="PCA on MNIST", layout="wide")
st.title("🔢  PCA on MNIST via SVD")
st.caption(
    "PCA = SVD of centered data. We compute "
    r"$X_c = U\Sigma V^\top$ and project onto top components."
)


# --------------------------------------------------------------------
# 1. Load data (cached)
# --------------------------------------------------------------------
@st.cache_data(show_spinner="Loading MNIST and computing SVD...")
def load_and_decompose(n_samples=10000, seed=0):
    import gzip
    import os
    import urllib.request

    BASE = "https://storage.googleapis.com/cvdf-datasets/mnist/"
    files = {
        "X": "train-images-idx3-ubyte.gz",
        "y": "train-labels-idx1-ubyte.gz",
    }
    cache_dir = "/tmp/mnist"
    os.makedirs(cache_dir, exist_ok=True)
    for fname in files.values():
        path = os.path.join(cache_dir, fname)
        if not os.path.exists(path):
            urllib.request.urlretrieve(BASE + fname, path)

    with gzip.open(os.path.join(cache_dir, files["X"]), "rb") as f:
        X_full = np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1, 784).astype(float)
    with gzip.open(os.path.join(cache_dir, files["y"]), "rb") as f:
        y_full = np.frombuffer(f.read(), np.uint8, offset=8).astype(int)

    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X_full), size=n_samples, replace=False)
    X = X_full[idx]
    y = y_full[idx]

    mean = X.mean(axis=0)
    Xc = X - mean
    U, S, Vt = svd(Xc, full_matrices=False)
    return X, Xc, y, mean, U, S, Vt


N = st.sidebar.select_slider(
    "Sample size", options=[2000, 5000, 10000, 20000], value=10000
)
X, Xc, y, mean_image, U, S, Vt = load_and_decompose(N)
n, d = X.shape
total_var = (S ** 2).sum() / (n - 1)
explained = (S ** 2 / (n - 1)) / total_var
cum_explained = np.cumsum(explained)


# --------------------------------------------------------------------
# 2. Sidebar controls
# --------------------------------------------------------------------
k = st.sidebar.slider("Number of components k", 2, 200, value=50)
digit_choice = st.sidebar.multiselect(
    "Digits to show in scatter", list(range(10)), default=list(range(10))
)


# --------------------------------------------------------------------
# 3. Tabs
# --------------------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs(
    ["📊  Variance", "🎯  2-D scatter", "👁️  Eigendigits", "🔄  Reconstruction"]
)

# -- Tab 1: explained variance --------------------------------------
with tab1:
    c1, c2, c3 = st.columns(3)
    c1.metric(f"Variance @ k={k}", f"{cum_explained[k-1]*100:.2f}%")
    c2.metric("Components for 90%", int(np.searchsorted(cum_explained, 0.9) + 1))
    c3.metric("Components for 95%", int(np.searchsorted(cum_explained, 0.95) + 1))

    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].plot(explained[:200], lw=1.5)
    ax[0].axvline(k, color="r", ls="--", alpha=0.7, label=f"k={k}")
    ax[0].set_xlabel("component index")
    ax[0].set_ylabel("variance ratio")
    ax[0].set_title("Per-component explained variance")
    ax[0].grid(alpha=0.3); ax[0].legend()

    ax[1].plot(cum_explained[:200], lw=2, color="crimson")
    ax[1].axvline(k, color="b", ls="--", alpha=0.7, label=f"k={k}")
    ax[1].axhline(0.9, color="k", ls=":", alpha=0.4, label="90%")
    ax[1].axhline(0.95, color="k", ls=":", alpha=0.4, label="95%")
    ax[1].set_xlabel("# components")
    ax[1].set_ylabel("cumulative")
    ax[1].set_title("Cumulative explained variance")
    ax[1].grid(alpha=0.3); ax[1].legend()

    st.pyplot(fig)


# -- Tab 2: 2-D scatter ---------------------------------------------
with tab2:
    pc_x = st.selectbox("X-axis component", list(range(1, 11)), index=0)
    pc_y = st.selectbox("Y-axis component", list(range(1, 11)), index=1)

    PC = U[:, [pc_x - 1, pc_y - 1]] * S[[pc_x - 1, pc_y - 1]]

    fig, ax = plt.subplots(figsize=(8, 6))
    cmap = plt.get_cmap("tab10")
    for digit in digit_choice:
        m = y == digit
        ax.scatter(PC[m, 0], PC[m, 1], s=4, alpha=0.5,
                   color=cmap(digit), label=str(digit))
    ax.set_xlabel(f"PC {pc_x}")
    ax.set_ylabel(f"PC {pc_y}")
    ax.set_title(f"MNIST projected onto PC {pc_x} vs PC {pc_y}")
    ax.legend(markerscale=2.5, ncol=2, loc="best")
    ax.grid(alpha=0.3)
    st.pyplot(fig)


# -- Tab 3: eigendigits ---------------------------------------------
with tab3:
    n_show = st.slider("How many eigendigits", 5, 30, value=15)
    cols = 5
    rows = (n_show + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))
    for i, ax in enumerate(axes.flat):
        if i < n_show:
            ax.imshow(Vt[i].reshape(28, 28), cmap="RdBu_r")
            ax.set_title(f"PC {i+1}", fontsize=9)
        ax.axis("off")
    plt.suptitle("Principal directions reshaped to 28×28")
    plt.tight_layout()
    st.pyplot(fig)


# -- Tab 4: reconstruction at chosen k ------------------------------
with tab4:
    st.write(f"Reconstructing using top **{k}** components.")

    rng = np.random.default_rng(42)
    sample_idx = rng.choice(n, size=6, replace=False)

    Vk = Vt[:k]
    coeffs = Xc[sample_idx] @ Vk.T
    reconstructed = coeffs @ Vk + mean_image

    fig, axes = plt.subplots(2, 6, figsize=(13, 4.5))
    for j, si in enumerate(sample_idx):
        axes[0, j].imshow(X[si].reshape(28, 28), cmap="gray")
        axes[0, j].set_title(f"orig (label {y[si]})", fontsize=9)
        axes[0, j].axis("off")
        axes[1, j].imshow(reconstructed[j].reshape(28, 28), cmap="gray")
        axes[1, j].set_title(f"rec @ k={k}", fontsize=9)
        axes[1, j].axis("off")
    plt.tight_layout()
    st.pyplot(fig)

    rel_err_sq = 1 - cum_explained[k - 1]
    st.info(
        f"Average squared relative reconstruction error at k={k}: "
        f"**{rel_err_sq:.4f}** "
        f"(by Eckart–Young this equals the trailing variance ratio)."
    )

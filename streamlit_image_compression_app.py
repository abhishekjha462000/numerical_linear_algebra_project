"""
Streamlit app: Interactive Image Compression via Truncated SVD.

Run with:
    streamlit run app_image_compression.py

Project 5 -- Low-Rank Approximation
Numerical Linear Algebra, MSc MMCS, University of Luxembourg
"""

import io

import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from PIL import Image
from numpy.linalg import svd, norm


st.set_page_config(page_title="SVD Image Compression", layout="wide")
st.title("🖼️  Image Compression via Truncated SVD")
st.caption(
    "Upload an image and adjust the rank $k$. We compute "
    r"$A = U \Sigma V^\top$ and reconstruct $A_k = U_k \Sigma_k V_k^\top$."
)


# --------------------------------------------------------------------
# 1. Load image
# --------------------------------------------------------------------
uploaded = st.sidebar.file_uploader(
    "Upload an image (PNG/JPG)", type=["png", "jpg", "jpeg"]
)

@st.cache_data
def default_image():
    """Fallback test image."""
    try:
        from skimage import data
        return data.camera().astype(float)
    except Exception:
        from scipy.datasets import ascent
        return ascent().astype(float)


if uploaded is None:
    st.sidebar.info("No image uploaded — using default test image.")
    img_gray = default_image()
    img_color = None
else:
    pil = Image.open(uploaded)
    img_color = np.array(pil.convert("RGB")).astype(float)
    img_gray = np.array(pil.convert("L")).astype(float)


mode = st.sidebar.radio(
    "Mode", ["Grayscale", "Color (per-channel SVD)"], index=0
)


# --------------------------------------------------------------------
# 2. SVD (cached so slider is instant)
# --------------------------------------------------------------------
@st.cache_data
def compute_svd(arr_bytes_key, arr):
    """Cached SVD. arr_bytes_key is just for hashing."""
    return svd(arr, full_matrices=False)


def arr_key(arr):
    return (arr.shape, float(arr.mean()), float(arr.std()))


if mode == "Grayscale":
    A = img_gray
    U, S, Vt = compute_svd(arr_key(A), A)
    max_k = len(S)
elif img_color is None:
    st.warning("Color mode requires an uploaded color image.")
    st.stop()
else:
    svds = []
    for c in range(3):
        svds.append(compute_svd(arr_key(img_color[..., c]), img_color[..., c]))
    max_k = min(len(s[1]) for s in svds)


# --------------------------------------------------------------------
# 3. Rank slider
# --------------------------------------------------------------------
k = st.sidebar.slider("Rank k", 1, int(max_k), value=min(20, int(max_k)))


def reconstruct_gray(U, S, Vt, k):
    return U[:, :k] @ np.diag(S[:k]) @ Vt[:k, :]


def reconstruct_color(svds, k):
    out = np.zeros((svds[0][0].shape[0], svds[0][2].shape[1], 3))
    for c in range(3):
        Uc, Sc, Vtc = svds[c]
        out[..., c] = reconstruct_gray(Uc, Sc, Vtc, k)
    return out


# --------------------------------------------------------------------
# 4. Display
# --------------------------------------------------------------------
col_left, col_right = st.columns(2)

if mode == "Grayscale":
    Ak = reconstruct_gray(U, S, Vt, k)
    with col_left:
        st.subheader("Original")
        st.image(np.clip(A, 0, 255).astype(np.uint8), use_container_width=True, clamp=True)
    with col_right:
        st.subheader(f"Rank-{k} reconstruction")
        st.image(np.clip(Ak, 0, 255).astype(np.uint8), use_container_width=True, clamp=True)

    # Metrics
    rel_err = norm(A - Ak, "fro") / norm(A, "fro")
    psnr = 20 * np.log10(255.0 / max(np.sqrt(np.mean((A - Ak) ** 2)), 1e-12))
    m, n = A.shape
    storage_ratio = k * (m + n + 1) / (m * n)

else:
    Ak = reconstruct_color(svds, k)
    with col_left:
        st.subheader("Original")
        st.image(np.clip(img_color, 0, 255).astype(np.uint8), use_container_width=True)
    with col_right:
        st.subheader(f"Rank-{k} reconstruction")
        st.image(np.clip(Ak, 0, 255).astype(np.uint8), use_container_width=True)

    rel_err = norm(img_color - Ak) / norm(img_color)
    psnr = 20 * np.log10(255.0 / max(np.sqrt(np.mean((img_color - Ak) ** 2)), 1e-12))
    m, n, _ = img_color.shape
    storage_ratio = 3 * k * (m + n + 1) / (3 * m * n)


# --------------------------------------------------------------------
# 5. Metrics
# --------------------------------------------------------------------
m1, m2, m3, m4 = st.columns(4)
m1.metric("Rank k", f"{k}")
m2.metric("Relative error", f"{rel_err:.4f}")
m3.metric("PSNR (dB)", f"{psnr:.2f}")
m4.metric("Storage ratio", f"{storage_ratio:.3f}",
          delta=f"{(1/storage_ratio):.1f}× compression",
          delta_color="off")


# --------------------------------------------------------------------
# 6. Singular value plot
# --------------------------------------------------------------------
st.markdown("---")
st.subheader("Singular value spectrum")

if mode == "Grayscale":
    S_plot = S
else:
    # average of three channels
    S_plot = np.mean([s[1] for s in svds], axis=0)

fig, ax = plt.subplots(1, 2, figsize=(12, 3.5))
ax[0].plot(S_plot, lw=1.3)
ax[0].axvline(k, color="r", ls="--", alpha=0.7, label=f"k={k}")
ax[0].set_xlabel("index $i$"); ax[0].set_ylabel(r"$\sigma_i$")
ax[0].set_title("Linear scale"); ax[0].grid(alpha=0.3); ax[0].legend()

ax[1].semilogy(S_plot, lw=1.3, color="crimson")
ax[1].axvline(k, color="b", ls="--", alpha=0.7, label=f"k={k}")
ax[1].set_xlabel("index $i$"); ax[1].set_ylabel(r"$\sigma_i$ (log)")
ax[1].set_title("Log scale"); ax[1].grid(alpha=0.3, which="both"); ax[1].legend()

st.pyplot(fig)


# --------------------------------------------------------------------
# 7. Theory reminder
# --------------------------------------------------------------------
with st.expander("📐  Theory recap"):
    st.markdown(
        r"""
**Eckart–Young–Mirsky.** For any matrix $B$ with $\mathrm{rank}(B)\leq k$,
$$
\|A - A_k\|_2 \leq \|A - B\|_2, \qquad \|A - A_k\|_F \leq \|A - B\|_F,
$$
with equalities
$$
\|A - A_k\|_2 = \sigma_{k+1}, \qquad \|A - A_k\|_F = \sqrt{\sum_{i>k}\sigma_i^2}.
$$
Truncated SVD is the *optimal* rank-$k$ approximation in both norms.

**Storage.** Full image: $mn$ values. Rank-$k$ store: $k(m+n+1)$ values.
"""
    )

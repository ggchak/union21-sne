"""Diagnósticos de convergência das cadeias.

Os traces e a taxa de aceitação são inspecionados graficamente no notebook;
aqui estão os diagnósticos numéricos:

* `tau_int`  — tempo de autocorrelação integrado, com janela de Sokal;
* `ess`      — tamanho efetivo de amostra N/τ;
* `mcse`     — erro de Monte Carlo da média, s/sqrt(ESS);
* `split_rhat` — R-chapéu com cada cadeia partida ao meio;
* `dunkley`  — ajuste P(k) = P0/[1+(k/k*)^α] e as flags j* > 20 e r < 0.01.

Todos são aplicados às cadeias retidas sem afinamento (*thinning*): afinar
descarta informação e melhora artificialmente a aparência da autocorrelação sem
melhorar a precisão das estimativas.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize


# ------------------------------------------------------------ autocorrelação

def autocorrelacao(x, lag_max: int | None = None) -> np.ndarray:
    """ρ_ℓ normalizada, por FFT (com laço em Python seria minutos, não ms)."""
    x = np.asarray(x, dtype=float)
    y = x - x.mean()
    n_fft = 1 << int(np.ceil(np.log2(2 * x.size)))
    f = np.fft.rfft(y, n_fft)
    acf = np.fft.irfft(f * np.conjugate(f), n_fft)[:x.size].real
    acf = acf / acf[0] if acf[0] != 0 else acf
    return acf if lag_max is None else acf[:lag_max]


def tau_int(x, c: float = 5.0) -> float:
    """τ_int = 1 + 2 Σ ρ_ℓ, truncado pela janela automática de Sokal."""
    taus = 2.0 * np.cumsum(autocorrelacao(x)) - 1.0
    janela = np.arange(taus.size)
    ok = janela < c * taus
    i = int(np.argmin(ok)) if np.any(~ok) else taus.size - 1
    return float(max(taus[i], 1.0))


def ess(x) -> float:
    """Tamanho efetivo de amostra: o comprimento da cadeia NÃO é o tamanho."""
    return float(np.asarray(x).size / tau_int(x))


def mcse(x) -> float:
    """Erro de Monte Carlo da média."""
    x = np.asarray(x, dtype=float)
    return float(np.std(x, ddof=1) / np.sqrt(ess(x)))


# ------------------------------------------------------------------ R-chapéu

def split_rhat(cadeias) -> float:
    """R-chapéu dividido para um parâmetro. `cadeias`: array (m, n)."""
    c = np.atleast_2d(np.asarray(cadeias, dtype=float))
    metade = c.shape[1] // 2
    c = np.concatenate([c[:, :metade], c[:, metade:2 * metade]], axis=0)
    m, n = c.shape
    if n < 4 or m < 2:
        return np.nan
    W = float(c.var(axis=1, ddof=1).mean())          # variância intra-cadeia
    B = float(n * c.mean(axis=1).var(ddof=1))        # variância entre cadeias
    if W <= 0:
        return np.nan
    return float(np.sqrt(((n - 1) / n * W + B / n) / W))


# ------------------------------------------------- espectro (Dunkley et al.)

@dataclass
class Dunkley:
    P0: float
    k_estrela: float
    alpha: float
    j_estrela: float
    r: float

    @property
    def passa(self) -> bool:
        return bool(self.j_estrela > 20.0 and self.r < 0.01)


def espectro(x):
    """Periodograma da cadeia centrada: devolve (j, k, P) para j = 1...N/2.

    a_j = (1/sqrt(N)) Σ (x_n - x̄) e^{2πijn/N},  P_j = |a_j|²,  k_j = 2πj/N.
    """
    x = np.asarray(x, dtype=float)
    a = np.fft.rfft(x - x.mean()) / np.sqrt(x.size)
    P = np.abs(a) ** 2
    j = np.arange(P.size)
    return j[1:], 2.0 * np.pi * j[1:] / x.size, P[1:]


def modelo_dunkley(k, P0, k_estrela, alpha):
    return P0 / (1.0 + (k / k_estrela) ** alpha)


def ajustar_dunkley(x, n_iter: int = 3) -> Dunkley:
    """Ajusta P(k) = P0/[1+(k/k*)^α] ao espectro da cadeia.

    O periodograma é *exponencialmente* distribuído em torno do espectro
    verdadeiro, então o ajuste correto é por máxima verossimilhança,
    minimizando Σ [ln P(k_j) + P̂_j/P(k_j)] — e não mínimos quadrados em log P,
    que seria enviesado. A faixa ajustada é escolhida iterativamente como
    j <= 10 j*, como no artigo original.
    """
    x = np.asarray(x, dtype=float)
    n = x.size
    j, k, P = espectro(x)
    s2 = float(np.var(x, ddof=1))
    j_max = int(np.clip(max(200, n // 10), 20, j.size))
    theta = np.array([np.log(max(P[:10].mean(), 1e-300)),
                      np.log(2 * np.pi * 20 / n), np.log(2.0)])

    for _ in range(n_iter):
        kk, PP = k[:j_max], P[:j_max]

        def nll(p):
            mod = np.maximum(modelo_dunkley(kk, *np.exp(p)), 1e-300)
            return float(np.sum(np.log(mod) + PP / mod))

        theta = minimize(nll, theta, method="Nelder-Mead",
                         options={"maxiter": 20000, "fatol": 1e-8}).x
        j_estrela = np.exp(theta[1]) * n / (2 * np.pi)
        novo = int(np.clip(10 * j_estrela, 20, j.size))
        if novo == j_max:
            break
        j_max = novo

    P0, k_estrela, alpha = np.exp(theta)
    return Dunkley(P0=float(P0), k_estrela=float(k_estrela), alpha=float(alpha),
                   j_estrela=float(k_estrela * n / (2 * np.pi)),
                   r=float(P0 / (n * s2)))


# ------------------------------------------------------------------ relatório

def relatorio(cadeias, nomes) -> str:
    """Tabela de convergência. `cadeias`: array (n_cadeias, n_passos, n_par)."""
    arr = np.asarray(cadeias, dtype=float)
    m, n, d = arr.shape
    linhas = [f"Cadeias: {m} × {n} passos retidos (sem thinning)", "",
              f"{'parâmetro':>12} {'média':>9} {'desvio':>9} {'MCSE':>9} "
              f"{'split-R̂':>9} {'ESS':>9} {'j*':>8} {'r':>9}  situação"]
    for i, nome in enumerate(nomes):
        por_cadeia = arr[:, :, i]
        todos = por_cadeia.ravel()
        ess_total = sum(ess(por_cadeia[c]) for c in range(m))
        rhat = split_rhat(por_cadeia)
        dk = [ajustar_dunkley(por_cadeia[c]) for c in range(m)]
        j_est = min(f.j_estrela for f in dk)
        r = max(f.r for f in dk)
        situacao = ("ok" if rhat < 1.01 and all(f.passa for f in dk)
                    else "ATENÇÃO: rodar mais")
        linhas.append(
            f"{nome:>12} {todos.mean():>9.4f} {todos.std(ddof=1):>9.4f} "
            f"{todos.std(ddof=1) / np.sqrt(ess_total):>9.5f} {rhat:>9.4f} "
            f"{ess_total:>9.0f} {j_est:>8.0f} {r:>9.5f}  {situacao}")
    linhas += ["", "Critérios: split-R̂ < 1.01 (guia, não garantia); "
                   "j* > 20 e r < 0.01 (Dunkley et al. 2005)."]
    return "\n".join(linhas)

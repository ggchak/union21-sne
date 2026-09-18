"""Dados do Union2.1, verossimilhança gaussiana e amostrador Metropolis-Hastings.

Verossimilhança
---------------
As SNe Ia medem a forma da relação distância-redshift a menos de um zero-point
aditivo desconhecido,

    μ_th(z) = μ_forma(z; Ω_m, Ω_Λ) + M,      M = M_B - 5 log₁₀ h + cte,

que absorve a magnitude absoluta das SNe Ia e a constante de Hubble. Com
Δ = μ_obs - μ_forma e 1 = (1,...,1)ᵀ,

    χ²(M) = A - 2MB + M²D,
    A = Δᵀ C⁻¹ Δ,   B = Δᵀ C⁻¹ 1,   D = 1ᵀ C⁻¹ 1.

Adota-se prior plano para M; a integral em M é gaussiana e resulta em

    χ²_marg = A - B²/D + ln(D/2π),

expressão de Goliath et al. (2001), também usada no Apêndice C de Amanullah et
al. (2010). Como C é fixa, D e C⁻¹1 são constantes e ficam pré-calculados: cada
avaliação da verossimilhança custa um único `cho_solve`. A matriz C é fatorada
por Cholesky uma única vez e C⁻¹ nunca é formada.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.linalg import cho_factor, cho_solve

from .cosmologia import FLRW

ARQUIVO_MU = "SCPUnion2.1_mu_vs_z.txt"
ARQUIVO_COV = "SCPUnion2.1_covmat_sys.txt"      # covariância com sistemáticas


@dataclass
class Dados:
    """Vetor de módulos de distância e sua covariância, validados e fatorados."""

    nome: np.ndarray
    z: np.ndarray
    mu: np.ndarray
    sigma: np.ndarray          # erro estatístico tabelado (usado só em gráficos)
    cov: np.ndarray
    rotulo: str = "com sistemáticas"

    def __post_init__(self):
        n = self.z.size
        if not (self.mu.size == self.sigma.size == n):
            raise ValueError("z, mu e sigma têm tamanhos diferentes")
        if self.cov.shape != (n, n):
            raise ValueError(f"covariância {self.cov.shape} incompatível com {n} SNe")
        if not np.allclose(self.cov, self.cov.T, atol=1e-12):
            raise ValueError("a matriz de covariância não é simétrica")
        if not np.all(np.isfinite(self.cov)) or not np.all(np.isfinite(self.mu)):
            raise ValueError("há valores não finitos nos dados ou na covariância")
        # cho_factor levanta LinAlgError se C não for positiva definida.
        self.chol = cho_factor(self.cov, lower=True)

    @property
    def n(self) -> int:
        return int(self.z.size)

    def resolve(self, b):
        """Resolve C x = b por substituição triangular (sem formar C⁻¹)."""
        return cho_solve(self.chol, b)

    @property
    def log_det_cov(self) -> float:
        c, _ = self.chol
        return float(2.0 * np.sum(np.log(np.abs(np.diag(c)))))

    def resumo(self) -> str:
        fora = self.cov - np.diag(np.diag(self.cov))
        return (f"Union2.1 [{self.rotulo}]: N = {self.n} SNe, "
                f"z ∈ [{self.z.min():.3f}, {self.z.max():.3f}], "
                f"elementos fora da diagonal não nulos: "
                f"{np.mean(np.abs(fora) > 0):.1%}, "
                f"correlação máxima: {self.correlacao_maxima:.3f}")

    @property
    def correlacao_maxima(self) -> float:
        d = np.sqrt(np.diag(self.cov))
        corr = self.cov / np.outer(d, d) - np.eye(self.n)
        return float(np.max(np.abs(corr)))


def carregar(pasta="dados", arquivo_mu: str = ARQUIVO_MU,
             arquivo_cov: str = ARQUIVO_COV) -> Dados:
    """Lê a tabela μ(z) e a matriz de covariância com sistemáticas.

    A tabela ASCII do release tem cinco linhas de cabeçalho comentadas e cinco
    colunas separadas por tabulação: nome da SN, redshift, módulo de distância,
    erro do módulo de distância e a probabilidade de a hospedeira ter baixa
    massa. A matriz é 580 × 580 e sua diagonal já contém as variâncias.
    """
    pasta = Path(pasta)
    caminho_cov = Path(arquivo_cov)
    if not caminho_cov.is_absolute():
        caminho_cov = pasta / caminho_cov
    if not caminho_cov.exists():
        raise FileNotFoundError(
            f"{caminho_cov} não encontrado. A matriz de covariância com "
            "sistemáticas faz parte do release Union2.1 "
            "(http://supernova.lbl.gov/union/).")

    tabela = np.genfromtxt(pasta / arquivo_mu, dtype=None, encoding="utf-8",
                           names=["nome", "z", "mu", "sigma", "p_baixa_massa"])
    return Dados(nome=np.asarray(tabela["nome"], dtype=str),
                 z=np.asarray(tabela["z"], dtype=float),
                 mu=np.asarray(tabela["mu"], dtype=float),
                 sigma=np.asarray(tabela["sigma"], dtype=float),
                 cov=np.atleast_2d(np.loadtxt(caminho_cov)))


class Verossimilhanca:
    """log L(Ω_m, Ω_Λ) com o zero-point M marginalizado analiticamente.

    O valor de `H0_fiducial` é arbitrário: alterá-lo desloca μ_forma por uma
    constante, absorvida por M. Em consequência, as SNe Ia sozinhas não vinculam
    H0 nesta análise.
    """

    def __init__(self, dados: Dados, H0_fiducial: float = 70.0):
        self.dados = dados
        self.H0 = H0_fiducial
        um = np.ones(dados.n)
        self._Cinv_um = dados.resolve(um)
        self._D = float(um @ self._Cinv_um)
        self._z_max = float(dados.z.max())

    # -------------------------------------------------------------- modelo
    def mu_forma(self, om: float, ol: float):
        """μ teórico com H0 fiducial e M = 0; None se a cosmologia é inválida."""
        cosmo = FLRW(self.H0, om, ol)
        if not cosmo.valida(self._z_max):
            return None
        mu = cosmo.mu(self.dados.z)
        return mu if np.all(np.isfinite(mu)) else None

    def _ABD(self, om: float, ol: float):
        mu = self.mu_forma(om, ol)
        if mu is None:
            return None
        delta = self.dados.mu - mu
        A = float(delta @ self.dados.resolve(delta))
        B = float(delta @ self._Cinv_um)
        return A, B, self._D

    # --------------------------------------------------------------- χ² / L
    def chi2(self, om: float, ol: float) -> float:
        """χ² com M marginalizado analiticamente."""
        abd = self._ABD(om, ol)
        if abd is None:
            return np.inf
        A, B, D = abd
        return A - B * B / D + np.log(D / (2 * np.pi))

    def chi2_com_M(self, om: float, ol: float, M: float) -> float:
        """χ² com o zero-point fixado em M."""
        mu = self.mu_forma(om, ol)
        if mu is None:
            return np.inf
        d = self.dados.mu - mu - M
        return float(d @ self.dados.resolve(d))

    def chi2_perfilado(self, om: float, ol: float) -> float:
        """χ² minimizado em M (A - B²/D); usado para bondade de ajuste."""
        abd = self._ABD(om, ol)
        if abd is None:
            return np.inf
        A, B, D = abd
        return A - B * B / D

    def M_ajustado(self, om: float, ol: float) -> float:
        """Zero-point que minimiza o χ²: B/D."""
        abd = self._ABD(om, ol)
        return np.nan if abd is None else abd[1] / abd[2]

    @property
    def sigma_M(self) -> float:
        """Largura do posterior condicional de M: 1/sqrt(D)."""
        return float(1.0 / np.sqrt(self._D))

    @property
    def dof(self) -> int:
        """N menos dois parâmetros cosmológicos e o zero-point."""
        return self.dados.n - 3

    def log_like(self, theta) -> float:
        chi2 = self.chi2(float(theta[0]), float(theta[1]))
        return -np.inf if not np.isfinite(chi2) else -0.5 * chi2


# --------------------------------------------------------- prior e posterior

LIMITES = np.array([[0.0, 2.0],      # Ω_m
                    [-1.0, 3.0]])    # Ω_Λ


def log_prior(theta) -> float:
    """Prior plano e próprio na caixa 0 ≤ Ω_m ≤ 2, -1 ≤ Ω_Λ ≤ 3."""
    theta = np.asarray(theta, dtype=float)
    dentro = np.all((theta >= LIMITES[:, 0]) & (theta <= LIMITES[:, 1]))
    return -np.log(np.prod(LIMITES[:, 1] - LIMITES[:, 0])) if dentro else -np.inf


def construir_posterior(veros: Verossimilhanca):
    """log π(θ) = log L(θ) + log p(θ), com -inf fora do suporte.

    O prior é avaliado primeiro: pontos fora da caixa não chegam a custar uma
    avaliação da verossimilhança.
    """
    def log_posterior(theta) -> float:
        lp = log_prior(theta)
        if not np.isfinite(lp):
            return -np.inf
        ll = veros.log_like(theta)
        return -np.inf if not np.isfinite(ll) else lp + ll
    return log_posterior


# ------------------------------------------------------ Metropolis-Hastings

@dataclass
class Cadeia:
    amostras: np.ndarray        # (n_passos, ndim), incluindo estados repetidos
    log_prob: np.ndarray
    aceitos: np.ndarray

    @property
    def taxa_aceitacao(self) -> float:
        return float(np.mean(self.aceitos))


def metropolis(log_alvo, x0, n_passos: int, cov_proposta, rng) -> Cadeia:
    """Metropolis com passeio aleatório gaussiano de covariância fixa.

    A proposta θ' = θ + Lz, com LLᵀ = Σ_q e z ~ N(0, I), é simétrica, de modo que
    o quociente de propostas cancela e a decisão usa apenas a razão de densidades
    do alvo. Toda a aritmética é feita em log-probabilidade; alvos não finitos
    (fora do prior, E²(z) < 0, NaN) contam como rejeição. Em uma rejeição o
    estado atual é gravado novamente — descartar repetições alteraria as
    frequências amostradas — e o log-alvo do estado atual é mantido em cache.
    """
    x = np.atleast_1d(np.asarray(x0, dtype=float)).copy()
    logp = float(log_alvo(x))
    if not np.isfinite(logp):
        raise ValueError(f"estado inicial inválido em θ = {x}")

    L = np.linalg.cholesky(np.atleast_2d(cov_proposta))
    amostras = np.empty((n_passos, x.size))
    log_probs = np.empty(n_passos)
    aceitos = np.zeros(n_passos, dtype=bool)

    for t in range(n_passos):
        candidato = x + L @ rng.standard_normal(x.size)
        logp_novo = float(log_alvo(candidato))
        log_alpha = logp_novo - logp
        if np.isfinite(log_alpha) and np.log(rng.random()) < min(0.0, log_alpha):
            x, logp = candidato, logp_novo
            aceitos[t] = True
        amostras[t], log_probs[t] = x, logp

    return Cadeia(amostras=amostras, log_prob=log_probs, aceitos=aceitos)


def ajustar_proposta(log_alvo, x0, rng, cov_inicial=None, n_piloto: int = 3000):
    """Aquecimento: estima a covariância do posterior e devolve a proposta final.

    Um piloto curto fornece a covariância empírica, escalada por 2.38²/d
    (Roberts, Gelman & Gilks). A proposta resultante é congelada antes da fase
    retida: adaptar durante a amostragem quebraria a propriedade de Markov da
    cadeia e a garantia de distribuição invariante.
    """
    x0 = np.atleast_1d(np.asarray(x0, dtype=float))
    d = x0.size
    cov = np.diag([0.01, 0.02]) if cov_inicial is None else np.atleast_2d(cov_inicial)
    piloto = metropolis(log_alvo, x0, n_piloto, cov, rng)
    amostras = piloto.amostras[n_piloto // 2:]
    cov_post = np.cov(amostras, rowvar=False) + 1e-12 * np.eye(d)
    return (2.38 ** 2 / d) * cov_post, piloto


def rodar_cadeias(log_alvo, inicios, n_passos: int, cov_proposta,
                  semente: int = 20240916) -> list[Cadeia]:
    """Uma cadeia por ponto inicial, com sementes derivadas de uma semente-mãe."""
    sementes = np.random.SeedSequence(semente).spawn(len(inicios))
    return [metropolis(log_alvo, x0, n_passos, cov_proposta,
                       np.random.default_rng(s))
            for x0, s in zip(inicios, sementes)]


def pontos_iniciais(log_alvo, n_cadeias: int, semente: int = 7,
                    separacao: float = 0.4) -> list[np.ndarray]:
    """Sorteia do prior pontos válidos e mutuamente separados."""
    rng = np.random.default_rng(semente)
    pontos: list[np.ndarray] = []
    while len(pontos) < n_cadeias:
        th = rng.uniform(LIMITES[:, 0], LIMITES[:, 1])
        if not np.isfinite(log_alvo(th)):
            continue
        if all(np.linalg.norm(th - p) > separacao for p in pontos):
            pontos.append(th)
    return pontos

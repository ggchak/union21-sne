"""Cosmologia de fundo FLRW (matéria + curvatura + Lambda), sem impor planura.

    E²(z) = Ω_m(1+z)³ + Ω_k(1+z)² + Ω_Λ,      Ω_k = 1 - Ω_m - Ω_Λ
    χ(z)  = c ∫₀^z dz'/H(z')
    D_M   = S_K[χ],   D_A = D_M/(1+z),   D_L = (1+z)D_M
    μ(z)  = 5 log₁₀(D_L/Mpc) + 25

Distâncias em Mpc, H0 em km/s/Mpc, tempos em Gyr.

A integral de χ usa Gauss-Legendre de ordem fixa, vetorizada em z: o integrando
1/E(z) é suave, 16 nós dão erro relativo ~1e-9 (há teste) e as 580 SNe saem numa
única chamada — é o que torna viável rodar 10⁵ passos de MCMC.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

C_LUZ = 299792.458                      # km/s
_GYR = 3.0856775814913673e19 / 3.1556952e16   # (km/s/Mpc)^-1 -> Gyr

_X, _W = np.polynomial.legendre.leggauss(16)
_X, _W = 0.5 * (_X + 1.0), 0.5 * _W     # nós e pesos mapeados para [0, 1]


@dataclass(frozen=True)
class FLRW:
    H0: float = 70.0
    omega_m: float = 0.3
    omega_lambda: float = 0.7

    @property
    def omega_k(self) -> float:
        return 1.0 - self.omega_m - self.omega_lambda

    @property
    def D_H(self) -> float:
        """Distância de Hubble c/H0, em Mpc."""
        return C_LUZ / self.H0

    @property
    def q0(self) -> float:
        """Parâmetro de desaceleração hoje."""
        return 0.5 * self.omega_m - self.omega_lambda

    # ------------------------------------------------------------ expansão
    def E2(self, z):
        z = np.asarray(z, dtype=float)
        return (self.omega_m * (1 + z) ** 3 + self.omega_k * (1 + z) ** 2
                + self.omega_lambda)

    def E(self, z):
        """E(z) = H(z)/H0; nan onde E² <= 0 (não há solução de expansão)."""
        e2 = self.E2(z)
        with np.errstate(invalid="ignore"):
            return np.where(e2 > 0, np.sqrt(np.where(e2 > 0, e2, 1.0)), np.nan)

    def H(self, z):
        return self.H0 * self.E(z)

    def valida(self, z_max: float, n: int = 400) -> bool:
        """E²(z) > 0 em todo [0, z_max]?

        Para Ω_Λ grande e Ω_m pequeno o universo tem "bounce" (não houve Big
        Bang) e as distâncias não existem: o ponto deve ser rejeitado. Checar só
        nos nós da quadratura não basta — E² pode mergulhar entre dois nós.
        """
        return bool(np.all(self.E2(np.linspace(0.0, z_max, n)) > 0.0))

    # ----------------------------------------------------------- distâncias
    def chi(self, z):
        """Distância comóvel radial, em Mpc."""
        z = np.atleast_1d(np.asarray(z, dtype=float))
        e2 = self.E2(z[:, None] * _X[None, :])
        with np.errstate(invalid="ignore", divide="ignore"):
            f = np.where(e2 > 0, 1.0 / np.sqrt(np.where(e2 > 0, e2, 1.0)), np.nan)
        return self.D_H * z * (f * _W[None, :]).sum(axis=1)

    def D_M(self, z):
        """Distância comóvel transversal S_K[χ], em Mpc."""
        chi = self.chi(z)
        ok = self.omega_k
        if abs(ok) < 1e-8:                      # plano
            return chi
        raiz = np.sqrt(abs(ok))
        x = raiz * chi / self.D_H
        if ok > 0:                              # aberto
            return self.D_H / raiz * np.sinh(x)
        # fechado: além do antípoda (x >= pi) o seno decresce e a distância
        # deixa de fazer sentido -> rejeita
        return np.where(x < np.pi, self.D_H / raiz * np.sin(x), np.nan)

    def D_A(self, z):
        z = np.atleast_1d(np.asarray(z, dtype=float))
        return self.D_M(z) / (1.0 + z)

    def D_L(self, z):
        z = np.atleast_1d(np.asarray(z, dtype=float))
        return (1.0 + z) * self.D_M(z)

    def mu(self, z):
        """Módulo de distância; nan onde D_L <= 0."""
        d = self.D_L(z)
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(d > 0, 5.0 * np.log10(np.where(d > 0, d, 1.0)) + 25.0,
                            np.nan)

    # ------------------------------------------------------ tempo e volume
    def idade(self, z=0.0, n: int = 256):
        """Idade do universo em z, em Gyr (substituição u = 1/(1+z))."""
        z = np.atleast_1d(np.asarray(z, dtype=float))
        x, w = np.polynomial.legendre.leggauss(n)
        x, w = 0.5 * (x + 1.0), 0.5 * w
        u_max = 1.0 / (1.0 + z)
        uu = u_max[:, None] * x[None, :]
        e2 = self.E2(1.0 / uu - 1.0)
        with np.errstate(invalid="ignore", divide="ignore"):
            f = np.where(e2 > 0, 1.0 / (uu * np.sqrt(np.where(e2 > 0, e2, 1.0))),
                         np.nan)
        return _GYR / self.H0 * u_max * (f * w[None, :]).sum(axis=1)

    def tempo_conforme(self, z):
        """Tempo conforme percorrido desde z até hoje: η = χ/c, em Gyr."""
        return self.chi(z) / C_LUZ * _GYR

    def dV_dz_dOmega(self, z):
        """Elemento de volume comóvel c·D_M²/H(z), em Mpc³/sr."""
        return C_LUZ * self.D_M(z) ** 2 / self.H(z)

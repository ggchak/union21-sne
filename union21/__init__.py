"""union21 — inferência bayesiana de (Ω_m, Ω_Λ) com as SNe Ia do Union2.1.

    cosmologia    FLRW: E, H, χ, D_M, D_A, D_L, μ, idade, η, dV/(dz dΩ)
    analise       dados + verossimilhança (M marginalizado) + Metropolis-Hastings
    diagnosticos  τ_int, ESS, MCSE, split-R̂, espectro de Dunkley
"""

from .cosmologia import FLRW, C_LUZ
from .analise import (Dados, Verossimilhanca, Cadeia, LIMITES, carregar,
                      log_prior, construir_posterior, metropolis,
                      ajustar_proposta, rodar_cadeias, pontos_iniciais)
from . import diagnosticos

__version__ = "1.0.0"
__all__ = ["FLRW", "C_LUZ", "Dados", "Verossimilhanca", "Cadeia", "LIMITES",
           "carregar", "log_prior", "construir_posterior", "metropolis",
           "ajustar_proposta", "rodar_cadeias", "pontos_iniciais",
           "diagnosticos", "__version__"]

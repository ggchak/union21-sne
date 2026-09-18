"""Testes de aceitação do pacote union21.

Quatro blocos: cosmologia de fundo, núcleo bayesiano (amostrador), dados e
verossimilhança, e diagnósticos de convergência.

    pytest -q
"""

from pathlib import Path

import numpy as np
import pytest
from scipy.integrate import quad

from union21 import (C_LUZ, FLRW, Verossimilhanca, carregar, construir_posterior,
                     log_prior, metropolis)
from union21 import diagnosticos as dg

PASTA_DADOS = Path(__file__).resolve().parents[1] / "dados"


# ======================================================= 1. cosmologia

def test_E_em_z0_vale_um():
    for c in [FLRW(70, 0.3, 0.7), FLRW(70, 1.0, 0.0), FLRW(70, 0.2, 0.9)]:
        assert np.isclose(float(np.atleast_1d(c.E(0.0))[0]), 1.0)


def test_lei_de_hubble_a_baixo_z():
    """D_L → (cz/H0)[1 + (1-q0)z/2] quando z << 1."""
    c = FLRW(70, 0.3, 0.7)
    z = np.array([1e-4, 1e-3])
    esperado = C_LUZ * z / c.H0 * (1 + 0.5 * (1 - c.q0) * z)
    assert np.allclose(c.D_L(z), esperado, rtol=1e-6)


def test_D_M_igual_chi_no_plano():
    c = FLRW(70, 0.3, 0.7)                     # Ω_k = 0
    z = np.linspace(0.01, 1.5, 30)
    assert np.allclose(c.D_M(z), c.chi(z), rtol=1e-12)


def test_dualidade_de_distancia():
    """D_L = (1+z)² D_A, com ou sem curvatura (Etherington)."""
    for c in [FLRW(70, 0.3, 0.7), FLRW(70, 0.3, 0.2), FLRW(70, 0.5, 0.9)]:
        z = np.linspace(0.02, 1.4, 20)
        assert np.allclose(c.D_L(z), (1 + z) ** 2 * c.D_A(z), rtol=1e-12)


def test_quadratura_bate_com_quad():
    c = FLRW(70, 0.3, 0.7)
    for z in (0.05, 0.5, 1.414):
        ref = quad(lambda zz: 1.0 / float(np.atleast_1d(c.E(zz))[0]), 0, z,
                   epsabs=1e-12)[0] * c.D_H
        assert np.isclose(float(c.chi(z)[0]), ref, rtol=1e-9)


def test_universo_sem_big_bang_e_rejeitado():
    assert not FLRW(70, 0.1, 2.5).valida(1.5)        # E²(z) < 0 em algum z
    assert FLRW(70, 0.3, 0.7).valida(1.5)


def test_H0_so_desloca_mu_por_constante():
    """É essa degenerescência que o zero-point M absorve."""
    z = np.linspace(0.02, 1.4, 20)
    d = FLRW(50, 0.3, 0.7).mu(z) - FLRW(70, 0.3, 0.7).mu(z)
    assert np.allclose(d, d[0]) and np.isclose(d[0], 5 * np.log10(70 / 50))


def test_idade_do_universo():
    assert 13.0 < float(FLRW(70, 0.3, 0.7).idade(0.0)[0]) < 14.0


# ================================================== 2. núcleo bayesiano

MU, COV = np.array([1.0, -2.0]), np.array([[1.0, 0.8], [0.8, 2.0]])
COV_INV = np.linalg.inv(COV)


def alvo_gaussiano(theta):
    d = np.asarray(theta, float) - MU
    return float(-0.5 * d @ COV_INV @ d)


def test_recupera_gaussiana_conhecida():
    ch = metropolis(alvo_gaussiano, [0.0, 0.0], 40_000, COV * (2.38**2 / 2),
                    np.random.default_rng(1))
    x = ch.amostras[4_000:]
    erro = x.std(axis=0, ddof=1) / np.sqrt([dg.ess(x[:, i]) for i in range(2)])
    assert np.all(np.abs(x.mean(axis=0) - MU) < 4 * erro)
    assert np.allclose(np.cov(x, rowvar=False), COV, rtol=0.15, atol=0.15)


def test_mesma_semente_reproduz_a_cadeia():
    args = (alvo_gaussiano, [0.0, 0.0], 500, np.eye(2))
    a = metropolis(*args, np.random.default_rng(7))
    b = metropolis(*args, np.random.default_rng(7))
    c = metropolis(*args, np.random.default_rng(8))
    assert np.array_equal(a.amostras, b.amostras)
    assert not np.array_equal(a.amostras, c.amostras)


def test_rejeicao_repete_o_estado():
    ch = metropolis(alvo_gaussiano, [0.0, 0.0], 2_000, 9 * np.eye(2),
                    np.random.default_rng(3))
    assert not np.all(ch.aceitos)
    rej = np.where(~ch.aceitos[1:])[0] + 1
    assert np.all(ch.amostras[rej] == ch.amostras[rej - 1])
    assert len(ch.amostras) == 2_000            # nada é descartado


def test_prior_e_posterior_rejeitam_fora_do_suporte():
    assert np.isfinite(log_prior([0.3, 0.7]))
    assert log_prior([2.5, 0.7]) == -np.inf     # fora da caixa do prior
    assert log_prior([0.3, 3.5]) == -np.inf


def test_estado_inicial_invalido_levanta_erro():
    with pytest.raises(ValueError):
        metropolis(lambda t: -np.inf, [0.0, 0.0], 10, np.eye(2),
                   np.random.default_rng(0))


# ======================================== 3. dados e verossimilhança

@pytest.fixture(scope="module")
def dados():
    return carregar(PASTA_DADOS)


@pytest.fixture(scope="module")
def veros(dados):
    return Verossimilhanca(dados)


def test_dimensoes_validadas(dados):
    assert dados.n == 580 and dados.cov.shape == (580, 580)
    assert np.all(np.isfinite(dados.mu)) and dados.z.min() > 0


def test_diagonal_da_covariancia_contem_as_variancias(dados):
    """A diagonal já traz as variâncias; com sistemáticas, acima do erro tabelado."""
    desvio = np.sqrt(np.diag(dados.cov))
    assert np.all(desvio >= dados.sigma - 1e-8)
    assert np.any(desvio > 1.05 * dados.sigma)


def test_covariancia_tem_correlacoes_fora_da_diagonal(dados):
    """A matriz com sistemáticas correlaciona SNe de um mesmo levantamento."""
    fora = dados.cov - np.diag(np.diag(dados.cov))
    assert np.mean(np.abs(fora) > 0) > 0.5
    assert 0.1 < dados.correlacao_maxima < 1.0


def test_cholesky_bate_com_solve_direto(dados):
    b = np.random.default_rng(0).standard_normal(dados.n)
    assert np.allclose(dados.resolve(b), np.linalg.solve(dados.cov, b), rtol=1e-8)


def test_marginalizacao_analitica_bate_com_integral_numerica(veros):
    """χ²_marg = -2 ln ∫ L(M) dM, calculado numericamente."""
    om, ol = 0.3, 0.7
    M0, s = veros.M_ajustado(om, ol), veros.sigma_M
    chi2_min = veros.chi2_perfilado(om, ol)
    val, _ = quad(lambda M: np.exp(-0.5 * (veros.chi2_com_M(om, ol, M) - chi2_min)),
                  M0 - 40 * s, M0 + 40 * s, limit=400)
    assert np.isclose(veros.chi2(om, ol), chi2_min - 2 * np.log(val), rtol=1e-6)


def test_chi2_marginalizado_e_invariante(dados, veros):
    """Deslocar todo μ_obs, ou trocar o H0 fiducial, não muda nada."""
    deslocado = type(dados)(nome=dados.nome, z=dados.z, mu=dados.mu + 0.37,
                            sigma=dados.sigma, cov=dados.cov, rotulo="teste")
    assert np.isclose(veros.chi2(0.3, 0.7),
                      Verossimilhanca(deslocado).chi2(0.3, 0.7), rtol=1e-10)
    assert np.isclose(veros.chi2(0.3, 0.7),
                      Verossimilhanca(dados, H0_fiducial=55).chi2(0.3, 0.7),
                      rtol=1e-10)


def test_bondade_de_ajuste_e_aceleracao(veros):
    chi2 = veros.chi2_perfilado(0.28, 0.72)
    assert 0.8 < chi2 / veros.dof < 1.2                       # χ²/dof ≈ 1
    assert chi2 < veros.chi2_perfilado(1.0, 0.0)              # melhor que EdS


def test_posterior_rejeita_cosmologia_invalida(veros):
    post = construir_posterior(veros)
    assert post([0.1, 2.5]) == -np.inf          # sem Big Bang
    assert post([2.5, 0.7]) == -np.inf          # fora do prior
    assert np.isfinite(post([0.3, 0.7]))


# ====================================================== 4. diagnósticos

def ar1(n, rho, rng):
    x = np.empty(n)
    x[0] = rng.standard_normal() / np.sqrt(1 - rho**2)
    for i in range(1, n):
        x[i] = rho * x[i - 1] + rng.standard_normal()
    return x


def test_ess_de_amostras_independentes():
    x = np.random.default_rng(1).standard_normal(40_000)
    assert 0.9 < dg.tau_int(x) < 1.6 and dg.ess(x) > 0.6 * x.size


def test_tau_bate_com_a_teoria_de_ar1():
    x = ar1(150_000, 0.8, np.random.default_rng(2))
    assert np.isclose(dg.tau_int(x), (1 + 0.8) / (1 - 0.8), rtol=0.2)


def test_rhat_detecta_cadeia_fora_do_lugar():
    rng = np.random.default_rng(4)
    boas = rng.standard_normal((4, 10_000))
    assert dg.split_rhat(boas) < 1.01
    ruins = rng.standard_normal((4, 5_000))
    ruins[0] += 3.0
    assert dg.split_rhat(ruins) > 1.1


def test_dunkley_passa_em_ruido_branco_e_falha_em_cadeia_presa():
    x = np.random.default_rng(9).standard_normal(50_000)
    bom = dg.ajustar_dunkley(x)
    assert bom.passa and np.isclose(bom.r, 1 / x.size, rtol=0.6)
    ruim = dg.ajustar_dunkley(ar1(2_000, 0.995, np.random.default_rng(10)))
    assert not ruim.passa and ruim.alpha > 1.0      # inclinação de random walk

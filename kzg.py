import time, random

from py_ecc import optimized_bls12_381 as b

from roots_of_unity import get_roots_of_unity
from trusted_setup import SETUP_G1, SETUP_G2
import fft
import params
import polynomials
import util

MODULUS = b.curve_order

def create_kzg_commitment(polynomial):
    """Commit to polynomial in coefficient form"""
    return util.lincomb(SETUP_G1[:len(polynomial)], polynomial)

def blob_to_commitment(blob):
    """Interpolate a polynomial (coefficients) from a blob in reverse bit order"""
    degree = len(blob)
    assert util.is_power_of_two(degree)

    polynomial = fft.fft(util.list_to_reverse_bit_order(blob), MODULUS, get_roots_of_unity(degree), inv=True)
    return create_kzg_commitment(polynomial)

def create_kzg_proof(polynomial, x):
    """Create KZG proof for polynomial in coefficient form"""
    quotient_polynomial = polynomials.div_polys(polynomial, [-x, 1])
    return util.lincomb(SETUP_G1[:len(quotient_polynomial)], quotient_polynomial)

def verify_kzg_proof(commitment, x, y, proof):
    s_minus_x = b.add(SETUP_G2[1], b.multiply(b.neg(b.G2), x))
    commitment_minus_y = b.add(commitment, b.multiply(b.neg(b.G1), y))

    pairing_check = b.pairing(b.G2, b.neg(commitment_minus_y), False)
    pairing_check *= b.pairing(s_minus_x, proof, False)
    pairing = b.final_exponentiate(pairing_check)

    return pairing == b.FQ12.one()

def create_kzg_multiproof(polynomial, h, n):
    """
    Compute Kate proof for polynomial in coefficient form at positions h * w^y where w is
    an n-th root of unity (this is the proof for one data availability sample, which consists
    of several polynomial evaluations). `h` is the shifting factor of the coset.
    """
    zero_poly = [-pow(h, n, MODULUS)] + [0] * (n - 1) + [1]
    quotient_polynomial = polynomials.div_polys(polynomial, zero_poly)
    return util.lincomb(SETUP_G1[:len(quotient_polynomial)], quotient_polynomial)


def verify_kzg_multiproof(commitment, x, ys, proof):
    """
    Check a proof for a Kate commitment for an evaluation f(x w^i) = y_i
    """
    def div(x, y):
        return x * polynomials.inv(y) % MODULUS

    n = len(ys)
    assert util.is_power_of_two(n)

    # Interpolate at a coset. Note because it is a coset, not the subgroup, we have to multiply the
    # polynomial coefficients by x^i
    interpolation_polynomial = fft.fft(ys, MODULUS, get_roots_of_unity(n), inv=True)
    interpolation_polynomial = [div(c, pow(x, i, MODULUS)) for i, c in enumerate(interpolation_polynomial)]

    # Verify the pairing equation
    #
    # e([commitment - interpolation_polynomial(s)], [1]) = e([proof],  [s^n - x^n])
    #    equivalent to
    # e([commitment - interpolation_polynomial]^(-1), [1]) * e([proof],  [s^n - x^n]) = 1_T
    #

    xn_minus_yn = b.add(SETUP_G2[n], b.multiply(b.neg(b.G2), pow(x, n, MODULUS)))
    commitment_minus_interpolation = b.add(commitment, b.neg(util.lincomb(SETUP_G1[:len(interpolation_polynomial)], interpolation_polynomial)))
    pairing_check = b.pairing(b.G2, b.neg(commitment_minus_interpolation), False)
    pairing_check *= b.pairing(xn_minus_yn, proof, False)
    pairing = b.final_exponentiate(pairing_check)
    return pairing == b.FQ12.one()

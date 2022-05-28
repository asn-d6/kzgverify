import time, random

from py_ecc import optimized_bls12_381 as b

from roots_of_unity import ROOTS_OF_UNITY, get_roots_of_unity
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
    degree = len(blob)
    polynomial = fft.fft(blob, MODULUS, get_roots_of_unity(degree), inv=True)
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

def create_kzg_multiproof(polynomial, x, n):
    """
    Compute Kate proof for polynomial in coefficient form at positions x * w^y where w is
    an n-th root of unity (this is the proof for one data availability sample, which consists
    of several polynomial evaluations)
    """
    zero_poly = [-pow(x, n, MODULUS)] + [0] * (n - 1) + [1]
    quotient_polynomial = polynomials.div_polys(polynomial, zero_poly)
    return util.lincomb(SETUP_G1[:len(quotient_polynomial)], quotient_polynomial)


def verify_kzg_multiproof(commitment, x, ys, proof):
    """
    Check a proof for a Kate commitment for an evaluation f(x w^i) = y_i
    """
    def div(x, y):
        return x * polynomials.inv(y) % MODULUS

    n = len(ys)

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


time_cache = [time.time()]
def get_time_delta():
    time_cache.append(time.time())
    return time_cache[-1] - time_cache[-2]

if __name__ == "__main__":
    polynomial = [random.randint(0, MODULUS) for i in range(32)]
    n = len(polynomial)

    x = 17 # where to evaluate the poly

    commitment = create_kzg_commitment(polynomial)
    proof = create_kzg_proof(polynomial, x)
    print("created kzg proof: {:.3f}s".format(get_time_delta()))

    value = polynomials.evaluate_poly_in_coefficient_form(polynomial, x)
    assert verify_kzg_proof(commitment, 17, value, proof)
    print("verified kzg proof: {:.3f}s".format(get_time_delta()))

    assert not verify_kzg_proof(commitment, 18, value, proof)
    print("verified faulty kzg proof: {:.3f}s".format(get_time_delta()))

    # Test blob_to_commitment()
    evaluations = [polynomials.evaluate_poly_in_coefficient_form(polynomial, z) for z in get_roots_of_unity(32)]
    commitment2 = blob_to_commitment(evaluations)
    assert b.eq(commitment, commitment2)
    print("tested blob to commttment: {:.3f}s".format(get_time_delta()))

    multiproof = create_kzg_multiproof(polynomial, x, 16)
    print("created multiproof: {:.3f}s".format(get_time_delta()))

    omega = get_roots_of_unity(16)[1]
    coset = [x * pow(omega, i, MODULUS) for i in range(16)]
    ys = [polynomials.evaluate_poly_in_coefficient_form(polynomial, z) for z in coset]

    assert verify_kzg_multiproof(commitment, x, ys, multiproof)
    print("verified multiproof: {:.3f}s".format(get_time_delta()))

from py_ecc import optimized_bls12_381 as b

from roots_of_unity import ROOTS_OF_UNITY
from trusted_setup import SETUP_G1, SETUP_G2

import params
import polynomials
import util

MODULUS = b.curve_order

def create_kzg_commitment(polynomial):
    """Commit to polynomial in coefficient form"""
    return util.lincomb(SETUP_G1[:len(polynomial)], polynomial)

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

if __name__ == "__main__":
    polynomial = [1, 2, 3, 4, 7, 7, 7, 7, 13, 13, 13, 13, 13, 13, 13, 13]
    n = len(polynomial)

    x = 17 # where to evaluate the poly

    commitment = create_kzg_commitment(polynomial)
    proof = create_kzg_proof(polynomial, x)
    value = polynomials.evaluate_poly_in_coefficient_form(polynomial, x)

    assert verify_kzg_proof(commitment, 17, value, proof)
    print("Single point check passed")

    assert not verify_kzg_proof(commitment, 18, value, proof)
    print("Faulty single point check rejected")

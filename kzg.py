from py_ecc import optimized_bls12_381 as b

import polynomials
import util

MODULUS = b.curve_order

## see kzg_data_availability/kzg_proofs.py

def generate_setup(s, size):
    """
    Generate trusted setup, in coefficient form.
    For data availability we always need to compute the polynomials anyway, so it makes little sense to do things in Lagrange space
    """
    return (
        [b.multiply(b.G1, pow(s, i, MODULUS)) for i in range(size + 1)],
        [b.multiply(b.G2, pow(s, i, MODULUS)) for i in range(size + 1)],
    )

def commit_to_poly(polynomial, setup_G1):
    """Commit to polynomial in coefficient form"""
    return util.lincomb(setup_G1[:len(polynomial)], polynomial)

def create_kzg_proof(polynomial, x, setup_G1):
    """Create KZG proof for polynomial in coefficient form"""
    quotient_polynomial = polynomials.div_polys(polynomial, [-x, 1])
    return util.lincomb(setup_G1[:len(quotient_polynomial)], quotient_polynomial)

def verify_kzg_proof(commitment, x, y, proof, setup_G2):
    s_minus_x = b.add(setup_G2[1], b.multiply(b.neg(b.G2), x))
    commitment_minus_y = b.add(commitment, b.multiply(b.neg(b.G1), y))

    pairing_check = b.pairing(b.G2, b.neg(commitment_minus_y), False)
    pairing_check *= b.pairing(s_minus_x, proof, False)
    pairing = b.final_exponentiate(pairing_check)

    return pairing == b.FQ12.one()

if __name__ == "__main__":
    polynomial = [1, 2, 3, 4, 7, 7, 7, 7, 13, 13, 13, 13, 13, 13, 13, 13]
    n = len(polynomial)

    x = 17 # where to evaluate the poly

    setup_G1, setup_G2 = generate_setup(1927409816240961209460912649124, n)

    commitment = commit_to_poly(polynomial, setup_G1)
    proof = create_kzg_proof(polynomial, x, setup_G1)
    value = polynomials.evaluate_poly_in_coefficient_form(polynomial, x)

    assert verify_kzg_proof(commitment, 17, value, proof, setup_G2)
    print("Single point check passed")

    assert not verify_kzg_proof(commitment, 18, value, proof, setup_G2)
    print("Faulty single point check rejected")

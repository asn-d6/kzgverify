import unittest, time, random

from py_ecc import optimized_bls12_381 as b

import params
from kzg import create_kzg_proof, verify_kzg_proof, create_kzg_commitment, blob_to_commitment, create_kzg_multiproof, verify_kzg_multiproof
from polynomials import evaluate_poly_in_coefficient_form, interpolate_polynomial
from fft import fft
from sharding import BlobsMatrix
from roots_of_unity import ROOTS_OF_UNITY

MODULUS = b.curve_order

time_cache = [time.time()]
def get_time_delta():
    time_cache.append(time.time())
    return time_cache[-1] - time_cache[-2]

class TestKzg(unittest.TestCase):
    def test_single_proof(self):
        polynomial = [random.randint(0, MODULUS) for i in range(32)]
        n = len(polynomial)

        x = 17 # where to evaluate the poly

        commitment = create_kzg_commitment(polynomial)
        proof = create_kzg_proof(polynomial, x)
        print("created kzg proof: {:.3f}s".format(get_time_delta()))

        value = evaluate_poly_in_coefficient_form(polynomial, x)
        assert verify_kzg_proof(commitment, 17, value, proof)
        print("verified kzg proof: {:.3f}s".format(get_time_delta()))

        assert not verify_kzg_proof(commitment, 18, value, proof)
        print("verified faulty kzg proof: {:.3f}s".format(get_time_delta()))

    def test_blob_to_commitment(self):
        polynomial = [random.randint(0, MODULUS) for i in range(32)]
        commitment = create_kzg_commitment(polynomial)

        evaluations = [evaluate_poly_in_coefficient_form(polynomial, z) for z in ROOTS_OF_UNITY[:32]]
        commitment2 = blob_to_commitment(evaluations)
        assert b.eq(commitment, commitment2)
        print("tested blob to commttment: {:.3f}s".format(get_time_delta()))

    def test_multiproof(self):
        polynomial = [random.randint(0, MODULUS) for i in range(32)]
        commitment = create_kzg_commitment(polynomial)

        x = 255

        multiproof = create_kzg_multiproof(polynomial, x, 16)
        print("created multiproof: {:.3f}s".format(get_time_delta()))

        omega = ROOTS_OF_UNITY[:16][1]
        coset = [x * pow(omega, i, MODULUS) for i in range(16)]
        ys = [evaluate_poly_in_coefficient_form(polynomial, z) for z in coset]

        assert verify_kzg_multiproof(commitment, x, ys, multiproof)
        print("verified multiproof: {:.3f}s".format(get_time_delta()))

class TestPolynomials(unittest.TestCase):
    def test_interpolation(self):
        # Generate random blob
        blob = [random.randint(0, MODULUS) for i in range(params.FIELD_ELEMENTS_PER_BLOB)]
        print("generated random blob: {:.3f}s".format(get_time_delta()))

        # Do an IFFT to get coefficient form of the blob polynomial
        coefs = fft(blob, MODULUS, ROOTS_OF_UNITY, inv=True)
        print("got coeff form using IFFT: {:.3f}s".format(get_time_delta()))

        # Let's do `n` evaluations on the `xs` and make sure we can interpolate back
        xs = range(1, params.FIELD_ELEMENTS_PER_BLOB+1)
        evaluations = []
        for x in xs:
            y = evaluate_poly_in_coefficient_form(coefs, x)
            evaluations.append(y)
        print("performed evaluations in coeff form: {:.3f}s".format(get_time_delta()))

        # Check that the interpolated polynomial matches
        interpolated_poly = interpolate_polynomial(xs, evaluations)
        print("interpolated first: {:.3f}s".format(get_time_delta()))

        assert interpolated_poly == coefs

class TestSharding(unittest.TestCase):
    def test_sharding(self):
        bm = BlobsMatrix()
        print("created blobs matrix: {:.3f}s".format(get_time_delta()))

        sample, commitment = bm.get_random_sample()
        print("got random sample: {:.3f}s".format(get_time_delta()))

        assert sample.verify_multiproof(commitment)
        print("verified multiproof: {:.3f}s".format(get_time_delta()))

if __name__ == '__main__':
    unittest.main()


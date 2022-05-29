import os, random, time

from py_ecc import optimized_bls12_381 as b

from imported.fft import fft
from imported.kzg_proofs import is_power_of_two, get_root_of_unity, list_to_reverse_bit_order, commit_to_poly, compute_proof_multi, check_proof_multi

import util
import params
from trusted_setup import SETUP

MODULUS = b.curve_order

# Dimensions of our matrix
N_MATRIX_ROWS = 4
N_MATRIX_COLUMNS = 4

# Each cell of the matrix is a sample
SAMPLES_PER_BLOB = N_MATRIX_COLUMNS

class Sample(object):
    """
    Represents a DAS sample (16 field elements)

    A sample has a bunch of field elements and a multiproof
    """
    def __init__(self, sample_index, data, polynomial):
        assert len(data) == params.FIELD_ELEMENTS_PER_SAMPLE
        n_openings = params.FIELD_ELEMENTS_PER_SAMPLE

        self.sample_index = sample_index
        self.data_points = data

        shifting_factor = util.get_coset_factor(n_openings, sample_index)
        self.multiproof = compute_proof_multi(polynomial, shifting_factor, n_openings, SETUP)

    def verify_multiproof(self, commitment):
        """Verify multiproof given a commitment to this sample"""
        n_openings = params.FIELD_ELEMENTS_PER_SAMPLE

        shifting_factor = util.get_coset_factor(n_openings, self.sample_index)
        ys = list_to_reverse_bit_order(self.data_points)
        assert check_proof_multi(commitment, self.multiproof, shifting_factor, ys, SETUP)

        return True

class Blob(object):
    """
    Represents a blob (a row of the matrix)

    A blob has a bunch of samples and a commitment that corresponds to the polynomial
    """
    def __init__(self, data_points):
        """Get a blob from a bunch of data bytes"""
        assert len(data_points) == params.FIELD_ELEMENTS_PER_BLOB

        # Polynomial that corresponds to this blob
        polynomial = fft(list_to_reverse_bit_order(data_points), MODULUS, get_root_of_unity(len(data_points)), inv=True)

        # Split data into samples
        self.samples = [Sample(i, data_points[i*16:(i+1)*16], polynomial) for i in range(SAMPLES_PER_BLOB)]

        # Get commitment to polynomial
        self.commitment = commit_to_poly(polynomial, SETUP)

class BlobsMatrix(object):
    """
    Represents a sharding matrix of NxN dimensions and exposes a bunch of handy methods
    """
    def __init__(self):
        """Generate a random sharding matrix"""
        self.blobs = []

        for i in range(N_MATRIX_ROWS):
            data = [random.randint(0, MODULUS) for i in range(params.FIELD_ELEMENTS_PER_BLOB)]
            blob = Blob(data)
            self.blobs.append(blob)

    def _get_sample(self, r):
        n_row = r // SAMPLES_PER_BLOB
        n_column = r % SAMPLES_PER_BLOB
        print("get_sample(): got r=%d -> %d row %d column" % (r, n_row, n_column))

        blob = self.blobs[n_row]
        return blob.samples[n_column], blob.commitment

    def get_random_sample(self):
        n_total_samples = SAMPLES_PER_BLOB * N_MATRIX_ROWS
        r = random.randrange(0, n_total_samples)
        return self._get_sample(r)

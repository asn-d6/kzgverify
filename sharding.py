import os, random, time

from py_ecc import optimized_bls12_381 as b

import polynomials
import params
import fft
import kzg
from roots_of_unity import ROOTS_OF_UNITY

MODULUS = b.curve_order

# Dimensions of our matrix
N_MATRIX_ROWS = 4
N_MATRIX_COLUMNS = 4

# Each cell of the matrix is a sample
SAMPLES_PER_BLOB = N_MATRIX_COLUMNS
N_SAMPLES_TOTAL = SAMPLES_PER_BLOB * N_MATRIX_ROWS


class Sample(object):
    """
    Represents a DAS sample (16 field elements)

    A sample has a bunch of field elements and a multiproof
    """
    def __init__(self, sample_index, data, entire_row_data):
        assert len(data) == params.FIELD_ELEMENTS_PER_SAMPLE
        degree = params.FIELD_ELEMENTS_PER_SAMPLE

        self.sample_index = sample_index
        self.data_points = data

        row_polynomial = fft.fft(entire_row_data, MODULUS, ROOTS_OF_UNITY, inv=True)
        shifting_factor = ROOTS_OF_UNITY[degree*sample_index]
        self.multiproof = kzg.create_kzg_multiproof(row_polynomial, shifting_factor, degree)

    def verify_multiproof(self, commitment):
        """Verify multiproof given a commitment to this sample"""
        degree = params.FIELD_ELEMENTS_PER_SAMPLE

        shifting_factor = ROOTS_OF_UNITY[degree*self.sample_index]
        assert kzg.verify_kzg_multiproof(commitment, shifting_factor, self.data_points, self.multiproof)

class Blob(object):
    """
    Represents a blob (a row of the matrix)

    A blob has a bunch of samples and a commitment that corresponds to the polynomial
    """
    def __init__(self, data_points):
        """Get a blob from a bunch of data bytes"""
        assert len(data_points) == params.FIELD_ELEMENTS_PER_BLOB

        # Split data into samples
        self.samples = [Sample(i+1, data_points[i:i+16], data_points) for i in range(SAMPLES_PER_BLOB)]

        # Get commitment to polynomial
        self.commitment = kzg.blob_to_commitment(data_points)

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
        r = random.randrange(0, N_SAMPLES_TOTAL)
        return self._get_sample(r)

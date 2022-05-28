import os, random

from py_ecc import optimized_bls12_381 as b

import params
import kzg
from roots_of_unity import ROOTS_OF_UNITY

MODULUS = b.curve_order

DATA_BYTES_PER_BLOB = 31 * params.FIELD_ELEMENTS_PER_BLOB

# XXX no error correction atm
# Dimensions of our matrix
N_MATRIX_ROWS = 4
N_MATRIX_COLUMNS = 4

# Each column is a sample
SAMPLES_PER_BLOB = N_MATRIX_COLUMNS
# Each cell of the matrix is a sample
N_SAMPLES_TOTAL = SAMPLES_PER_BLOB * N_MATRIX_ROWS

class Sample(object):
    """
    Represents a DAS sample (16 field elements)

    A sample has a bunch of field elements and a multiproof
    """
    def __init__(self, data):
        assert len(data) == params.FIELD_ELEMENTS_PER_SAMPLE

        self.field_elements = data

        # Sanity check: they should be actual scalar field elements
        for field_element in self.field_elements:
            assert field_element < MODULUS

class Blob(object):
    """
    Represents a blob (a row of the matrix)

    A blob has a bunch of samples and a commitment that corresponds to the polynomial
    """
    def __init__(self, data):
        """Get a blob from a bunch of data bytes"""

        # Caller should have given us the right amount of data
        assert len(data) == DATA_BYTES_PER_BLOB

        # Split the data into field elements
        data_points = [int.from_bytes(data[i*31:(i+1)*31], byteorder='big') for i in range(params.FIELD_ELEMENTS_PER_BLOB)]
        self.samples = [Sample(data_points[i:i+16]) for i in range(SAMPLES_PER_BLOB)]

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
            data = os.urandom(DATA_BYTES_PER_BLOB)
            blob = Blob(data)
            self.blobs.append(blob)

    def _get_sample(self, r):
        n_row = r // SAMPLES_PER_BLOB
        n_column = r % SAMPLES_PER_BLOB

        print("get_sample(): got r=%d -> %d row %d column" % (r, n_row, n_column))

        blob = self.blobs[n_row]

        return blob.samples[n_column]

    def get_random_sample(self):
        r = random.randint(0, N_SAMPLES_TOTAL)
        return self._get_sample(r)

    def _get_row(self, r):
        return self.blobs[r].samples

    def get_random_row_samples(self):
        r = random.randint(0, N_MATRIX_ROWS)
        return self._get_row(r)

    def _get_column(self, r):
        return [blob.samples[r] for blob in self.blobs]

    def get_random_column_samples(self):
        r = random.randint(0, N_MATRIX_COLUMNS)
        return self._get_column(r)


if __name__ == '__main__':
    bm = BlobsMatrix()

    r = random.randint(0, N_SAMPLES_TOTAL)
    sample = bm._get_sample(r)

    n_row = r // SAMPLES_PER_BLOB
    n_column = r % SAMPLES_PER_BLOB

    row = bm._get_row(n_row)
    assert row[n_column] == sample

    column = bm._get_column(n_column)
    assert column[n_row] == sample


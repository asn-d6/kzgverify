import os

import params

from py_ecc import optimized_bls12_381 as b

DATA_BYTES_PER_BLOB = 31 * params.FIELD_ELEMENTS_PER_BLOB
FIELD_ELEMENTS_PER_SAMPLE = 16

# XXX no error correction atm
N_MATRIX_ROWS = 4
N_MATRIX_COLUMNS = 4

class Sample(object):
    """
    Represents a DAS sample (16 field elements)
    """
    def __init__(self, data):
        assert len(data) == FIELD_ELEMENTS_PER_SAMPLE
        self.field_elements = data

class Blob(object):
    """
    Represents a sharding blob (a row of the matrix)
    """
    def __init__(self, data):
        """Get a blob from a bunch of data bytes"""

        # Caller should have given us the right amount of data
        assert len(data) == DATA_BYTES_PER_BLOB

        # Start packing!
        self.blob = b''
        for i in range(params.FIELD_ELEMENTS_PER_BLOB):
            # Get the next 31 bytes from our data
            chunk = data[i*31:(i+1)*31]
            # Pad it to 32 bytes with a zero byte and stuff it into the blob
            self.blob += chunk + b'\x00'

        assert(len(self.blob) == 32*params.FIELD_ELEMENTS_PER_BLOB)

class ShardingMatrix(object):
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

if __name__ == '__main__':
    sm = ShardingMatrix()
    print(sm.blobs)

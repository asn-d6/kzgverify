import unittest, time, random

from py_ecc import optimized_bls12_381 as b

import util
import params
from sharding import BlobsMatrix

MODULUS = b.curve_order

time_cache = [time.time()]
def get_time_delta():
    time_cache.append(time.time())
    return time_cache[-1] - time_cache[-2]

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


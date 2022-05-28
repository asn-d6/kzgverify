from py_ecc import optimized_bls12_381 as b

import multicombs

def lincomb(points, scalars):
    return multicombs.lincomb(points, scalars, b.add, b.Z1)

def vector_lincomb(vectors, scalars):
    """
    Given a list of vectors, compute the linear combination of each column with `scalars`, and return the resulting
    vector.
    """
    r = [0]*len(vectors[0])
    for v, a in zip(vectors, scalars):
        for i, x in enumerate(v):
            r[i] = (r[i] + a * x) % MODULUS
    return [BLSFieldElement(x) for x in r]

def is_power_of_two(x):
    return x > 0 and x & (x-1) == 0

def reverse_bit_order(n, order):
    """
    Reverse the bit order of an integer n
    """
    assert is_power_of_two(order)
    # Convert n to binary with the same number of bits as "order" - 1, then reverse its bit order
    return int(('{:0' + str(order.bit_length() - 1) + 'b}').format(n)[::-1], 2)


def list_to_reverse_bit_order(l):
    """
    Convert a list between normal and reverse bit order. This operation is idempotent.
    """
    return [l[reverse_bit_order(i, len(l))] for i in range(len(l))]

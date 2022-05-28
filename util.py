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

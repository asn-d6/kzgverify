from py_ecc import optimized_bls12_381 as b

def lincomb(points, scalars):
    """
    BLS multiscalar multiplication. This function can be optimized using Pippenger's algorithm and variants.
    """
    assert len(points) == len(scalars)

    r = b.Z1
    for x, a in zip(points, scalars):
        r = b.add(r, b.multiply(x, a))
    return r

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

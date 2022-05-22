import random, time

import params
from fft import fft

MODULUS = 0x73eda753299d7d483339d80809a1d80553bda402fffe5bfeffffffff00000001
PRIMITIVE_ROOT_OF_UNITY = 7

def roots_of_unity(order):
    """
    Compute a list of roots of unity for a given order.
    The order must divide the BLS multiplicative group order, i.e. MODULUS - 1
    The order of the root of unity MUST divide phi(n) ^^
    """
    assert (MODULUS - 1) % order == 0
    roots = []
    root_of_unity = pow(PRIMITIVE_ROOT_OF_UNITY, (MODULUS - 1) // order, MODULUS)

    current_root_of_unity = 1
    for i in range(order):
        roots.append(current_root_of_unity)
        current_root_of_unity = current_root_of_unity * root_of_unity % MODULUS
    return roots

ROOTS_OF_UNITY = roots_of_unity(params.FIELD_ELEMENTS_PER_BLOB)

def inv(x):
    """
    Compute the modular inverse of x using the eGCD algorithm
    i.e. return y such that x * y % MODULUS == 1 and return 0 for x == 0
    """
    if x == 0:
        return 0

    lm, hm = 1, 0
    low, high = x % MODULUS, MODULUS
    while low > 1:
        r = high // low
        nm, new = hm - lm * r, high - low * r
        lm, low, hm, high = nm, new, lm, low
    return lm % MODULUS

def evaluate_poly_in_coefficient_form(poly, x):
    result = 0
    for i, coeff in enumerate(poly):
        result += coeff * (x**i)
    return result % MODULUS

def compute_powers(x, n):
    current_power = 1
    powers = []
    for _ in range(n):
        powers.append(BLSFieldElement(current_power))
        current_power = current_power * int(x) % MODULUS
    return powers

def div_polys(a, b):
    """Divide polynomials in coefficient form"""
    assert len(a) >= len(b)
    a = [x for x in a]
    o = []
    apos = len(a) - 1
    bpos = len(b) - 1
    diff = apos - bpos
    while diff >= 0:
        quot = a[apos] * inv(b[bpos]) % MODULUS
        o.insert(0, quot)
        for i in range(bpos, -1, -1):
            a[diff+i] -= b[i] * quot
        apos -= 1
        diff -= 1
    return [x % MODULUS for x in o]

def zpoly(xs):
    root = [1]
    for x in xs:
        root.insert(0, 0)
        for j in range(len(root)-1):
            root[j] -= root[j+1] * x
    return [x % MODULUS for x in root]

# Given p+1 y values and x values with no errors, recovers the original
# p+1 degree polynomial.
# Lagrange interpolation works roughly in the following way.
# 1. Suppose you have a set of points, eg. x = [1, 2, 3], y = [2, 5, 10]
# 2. For each x, generate a polynomial which equals its corresponding
#    y coordinate at that point and 0 at all other points provided.
# 3. Add these polynomials together.
def interpolate_polynomial(xs, ys):
    # Generate master numerator polynomial, eg. (x - x1) * (x - x2) * ... * (x - xn)
    root = zpoly(xs)
    assert len(root) == len(ys) + 1
    # print(root)
    # Generate per-value numerator polynomials, eg. for x=x2,
    # (x - x1) * (x - x3) * ... * (x - xn), by dividing the master
    # polynomial back by each x coordinate
    nums = [div_polys(root, [-x, 1]) for x in xs]
    # Generate denominators by evaluating numerator polys at each x
    denoms = [evaluate_poly_in_coefficient_form(nums[i], xs[i]) for i in range(len(xs))]
    invdenoms = [inv(denom) for denom in denoms]
    # Generate output polynomial, which is the sum of the per-value numerator
    # polynomials rescaled to have the right y values
    b = [0 for y in ys]
    for i in range(len(xs)):
        yslice = ys[i] * invdenoms[i] % MODULUS
        for j in range(len(ys)):
            if nums[i][j] and ys[i]:
                b[j] += nums[i][j] * yslice
    return [x % MODULUS for x in b]

time_cache = [time.time()]
def get_time_delta():
    time_cache.append(time.time())
    return time_cache[-1] - time_cache[-2]

if __name__ == '__main__':
    random.seed(1)

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

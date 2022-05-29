from py_ecc import optimized_bls12_381 as b

import random, time

import params
from fft import fft
from roots_of_unity import ROOTS_OF_UNITY

MODULUS = b.curve_order

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

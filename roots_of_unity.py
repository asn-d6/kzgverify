from py_ecc import optimized_bls12_381 as b

import params
import util

MODULUS = b.curve_order

PRIMITIVE_ROOT_OF_UNITY = 7

def get_roots_of_unity(order):
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

ROOTS_OF_UNITY = get_roots_of_unity(params.FIELD_ELEMENTS_PER_BLOB)
REVERSE_BIT_ORDER_ROOTS_OF_UNITY = util.list_to_reverse_bit_order(ROOTS_OF_UNITY)

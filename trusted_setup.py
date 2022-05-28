import json

from py_ecc import optimized_bls12_381 as b

MODULUS = b.curve_order

with open('trusted_setup_G1.json', 'r') as f:
    trusted_setup_G1 = json.load(f)

with open('trusted_setup_G2.json', 'r') as f:
    trusted_setup_G2 = json.load(f)

SETUP_G1 = []
for point in trusted_setup_G1["setup_G1"]:
    SETUP_G1.append((b.FQ(int(point[0])), b.FQ(int(point[1])), b.FQ.one()))

SETUP_G2 = []
for point in trusted_setup_G2['setup_G2']:
    SETUP_G2.append((b.FQ2((int(point[0][0]), int(point[0][1]))), b.FQ2((int(point[1][0]), int(point[1][1]))), b.FQ2.one()))

if __name__ == '__main__':
    def serialize_g1(point):
        return int(point)

    def serialize_g2(point):
        return int(point.coeffs[0]), int(point.coeffs[1])

    s = 7851823980 # INSECURE s

    setup_G1 = [b.multiply(b.G1, pow(s, i, MODULUS)) for i in range(8192 + 1)]
    setup_G1 = [b.normalize(pt) for pt in setup_G1]
    with open('trusted_setup_G1.json', 'w') as f:
        json.dump({"setup_G1" : setup_G1}, f, default=serialize_g1)

    setup_G2 = [b.multiply(b.G2, pow(s, i, MODULUS)) for i in range(8192 + 1)]
    setup_G2 = [b.normalize(pt) for pt in setup_G2]
    with open('trusted_setup_G2.json', 'w') as f:
        json.dump({"setup_G2" : setup_G2}, f, default=serialize_g2)

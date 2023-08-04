
config = dict(
    atoms=["O"] * 3 + ["H"] * 6,
    coordinates = [
        [-4.0793104723374253E-002,  -0.65008890541725539,      1.4801120395234866],
        [-3.9133966591103124E-002,  -0.95549559982249110,     -1.3051361607330014],
        [ 5.3176492244194275E-002,   1.6035938340842861,      -0.17335546537928878],

        [ 2.4159056687716802E-002,  -1.0888139777074795,       0.61898449568084435],
        [ 0.61821329070632103,      -1.0649910812517203,       2.0358410790461448],

        [-1.5658631743419457E-002,  1.0833818329912787E-002, -1.2322064627198799],
        [0.59883245798515161,      -1.1806918849913595,      -1.9815169675519309],

        [2.9296790475816546E-002,   1.0588876695068357,       0.62803315642787094],
        [-0.66936176962601512,      2.2249141511773813,      -8.3136807735043172E-002]
    ],
    walkers_per_core=1000,
    steps_per_call=5,
    parameters=dict(nwaters=3, imodel=3) #nwaters + imodel
)

# [
#         [-4.0793104723374253E-002,  -0.65008890541725539,      1.4801120395234866],
#         [-3.9133966591103124E-002,  -0.95549559982249110,     -1.3051361607330014],
#         [ 5.3176492244194275E-002,   1.6035938340842861,      -0.17335546537928878],
#
#         [ 2.4159056687716802E-002,  -1.0888139777074795,       0.61898449568084435],
#         [ 0.61821329070632103,      -1.0649910812517203,       2.0358410790461448],
#
#         [-1.5658631743419457E-002,  1.0833818329912787E-002, -1.2322064627198799],
#         [0.59883245798515161,      -1.1806918849913595,      -1.9815169675519309],
#
#         [2.9296790475816546E-002,   1.0588876695068357,       0.62803315642787094],
#         [-0.66936176962601512,      2.2249141511773813,      -8.3136807735043172E-002]
#     ]

# [
#         [1.58073,  0.328962,  -0.0899222],
#         [-0.842514, -0.268543, -1.35211],
#         [-0.728777, -0.0859986,   1.43847],
#
#         [0.879597, 0.159113,  -0.740181],
#         [2.30022, -0.250969,  -0.329905],
#
#         [-1.05736,  -0.281191, -0.405096],
#         [-1.56861,   0.18143,  -1.77757],
#
#         [-0.882999,  0.508252,    2.16891],
#         [0.178154,   0.0926357,   1.14076]
#     ]

# [
#  [ 25.3875809,       2.28446364,       8.01933861],
#  [ 22.9643402,       1.68695939,       6.75715494],
#  [ 23.0780773,       1.86950338,       9.54773140],
#  [ 24.6864510,       2.11461496,       7.36908007],
#  [ 26.1070786,       1.70453322,       7.77935553],
#  [ 22.7494984,       1.67431045,       7.70416498],
#  [ 22.2382431,       2.13693213,       6.33168697],
#  [ 22.9238548,       2.46375370,       10.2781725],
#  [ 23.9850082,       2.04813766,       9.25002480]
# ]

# [ 2.98714678,  0.62164809, -0.16992833],
# [-1.59212072, -0.50747272, -2.55511759],
# [-1.37718894, -0.1625138 ,  2.71831434],
#
# [ 1.66219743,  0.30067999, -1.39873937],
# [ 4.34678583, -0.47426268, -0.6234301 ],
#
# [-1.99812082, -0.53137398, -0.76552049],
# [-2.9642433 ,  0.34285301, -3.35912047],
#
# [-1.66862628,  0.96045708,  4.09864589],
# [ 0.33666227,  0.1750561 ,  2.15572397]
config  =  {
    'wrap_potential': False,
    'function_name': 'pot',
    'arguments': (('nwaters', 'int'), 'coords'),
    'requires_make': True,
    'linked_libs': ['mbpol'],
    'name': 'LegacyMBPol',
    'potential_source': '/config/potentials/LegacyMBPol/raw_source/LegacyMBPol',
    'static_source': False,
    # 'fortran_potential': False
    'raw_array_potential': True
}

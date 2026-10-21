// Molecular descriptors the spectrum predictor can use, in the order of
// fredis.domain.molecule.descriptors.SUPPORTED_DESCRIPTOR_NAMES.
module.exports = [
  'MolWt', 'ExactMolWt', 'HeavyAtomCount', 'TPSA', 'MolLogP', 'NumHAcceptors', 'NumHDonors',
  'NumRotatableBonds', 'RingCount', 'NumAromaticRings', 'NumAliphaticRings', 'FractionCSP3',
  'NumHeteroatoms', 'FormalCharge', 'BertzCT',
];

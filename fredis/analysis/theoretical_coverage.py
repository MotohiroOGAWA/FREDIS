"""Full Fragmenter assignments and observed-intensity coverage for spectrum pairs."""
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
import numpy as np

_fragmenter = None
_limits = None

def _init_theory(params, limits):
    global _fragmenter, _limits
    from fredis.domain.fragment.fragmenter import Fragmenter
    _fragmenter = Fragmenter.from_dict(params)
    _limits = limits


def _theory_group(task):
    """One chemistry tree per SMILES; reuse it across observed spectra/adducts."""
    from fredis.libs.mmkit.mmkit import Compound, Adduct
    smiles, records = task
    try:
        source = Compound.from_smiles(smiles)
        adducts = list(dict.fromkeys(Adduct.parse(row['adduct']) for row in records))
        tree = _fragmenter.build_fragment_ion_tree(source, precursor_filter=_fragmenter.precursor_filter(source, adducts), **_limits)
        results = []
        for row in records:
            try:
                adduct = Adduct.parse(row['adduct'])
                if not _fragmenter.resolve_precursor_reactions(tree, adduct):
                    raise ValueError('Precursor is unreachable under the Fragmenter settings.')
                annotations = _fragmenter.annotate_peaks(tree, adduct, row['mz'])
                results.append((row['pair'], [bool(matches) for matches in annotations], ''))
            except Exception as error:
                results.append((row['pair'], None, f'{type(error).__name__}: {error}'))
        return results
    except Exception as error:
        return [(row['pair'], None, f'{type(error).__name__}: {error}') for row in records]


def intensity_coverage(intensity, mask, alpha):
    """Same squared-vector convention as MSSIM: sum I**(2 alpha)."""
    values = np.asarray(intensity, dtype=float)
    if not len(values) or not np.any(values > 0):
        return 0.0
    weights = (values / values.max()) ** (2 * alpha)
    return float(weights[np.asarray(mask, dtype=bool)].sum() / weights.sum())



def calculate(observed, fragmenter_params, *, smiles_column='SMILES', adduct_column='AdductType',
              alpha=.5, workers=1, max_unique_fragment_smiles=-1,
              max_reaction_combinations=10000, progress=None):
    """One coverage/error per input spectrum; no prediction-path pruning."""
    limits={'max_unique_fragment_smiles':max_unique_fragment_smiles,
            'max_reaction_combinations':max_reaction_combinations}
    _init_theory(fragmenter_params,limits)
    tasks, intensities, errors, masks = {}, {}, {}, {}
    for pair in range(len(observed)):
        record=observed[pair]
        mz,intensity=record.peaks
        intensities[pair]=intensity
        smiles,adduct=record[smiles_column],record[adduct_column]
        if smiles is None or adduct is None or not str(smiles).strip() or not str(adduct).strip():
            errors[pair]='Missing SMILES or adduct.'
            continue
        tasks.setdefault(str(smiles),[]).append({'pair':pair,'adduct':str(adduct),'mz':mz.tolist()})
    def collect(results):
        for done,result in enumerate(results,1):
            for pair,mask,error in result:
                if error:errors[pair]=error
                else:masks[pair]=mask
            if progress:progress(done,len(tasks))
    if workers==1:
        collect(map(_theory_group,tasks.items()))
    elif tasks:
        with ProcessPoolExecutor(max_workers=min(workers,len(tasks)),mp_context=mp.get_context('spawn'),
                                 initializer=_init_theory,initargs=(fragmenter_params,limits)) as pool:
            collect(pool.map(_theory_group,tasks.items(),chunksize=1))
    return [intensity_coverage(intensities[i],masks[i],alpha) if i in masks else None
            for i in range(len(observed))], [errors.get(i,'') for i in range(len(observed))]


def enrich_file(path, similarity, observed, fragmenter_params, *, alpha, workers,
                smiles_column='SMILES', adduct_column='AdductType', mass_tolerance=None, progress=None):
    """Save standard MSSIM plus extension columns; preserve spectra and native metadata."""
    from pathlib import Path
    import json, os, tempfile
    import h5py
    target=Path(path)
    params=json.loads(json.dumps(fragmenter_params))
    if mass_tolerance:params['mass_tolerance']=mass_tolerance
    frame=similarity.to_pandas()
    with observed.select(frame['index2'].astype(int).tolist()) as selected:
        coverages,errors=calculate(selected,params,alpha=alpha,workers=workers,
                                  smiles_column=smiles_column,adduct_column=adduct_column,progress=progress)
    frame['theoretical_fragment_peak_intensity_coverage']=coverages
    frame['theoretical_fragment_error']=errors
    metadata=similarity.metadata
    metadata['fredis_theoretical_coverage']={'intensity_exponent':alpha,'observed_side':2,
        'coverage_definition':'sum(assigned observed I^(2*alpha)) / sum(all observed I^(2*alpha))',
        'fragmenter_params':params,'search_limits':{'max_unique_fragment_smiles':-1,'max_reaction_combinations':10000},'failure_count':sum(bool(e) for e in errors)}
    metadata['fredis_prediction']={'observed_side':2}
    similarity.metadata=metadata
    fd,name=tempfile.mkstemp(prefix='.theoretical-',suffix='.mssim',dir=target.parent)
    os.close(fd)
    temporary=Path(name)
    try:
        temporary.unlink()
        similarity.save(temporary)
        with h5py.File(temporary,'r+') as handle:
            del handle['table.parquet']
            handle.create_dataset('table.parquet',data=np.frombuffer(frame.to_parquet(index=False),dtype=np.uint8))
        os.replace(temporary,target)
    finally:temporary.unlink(missing_ok=True)
    return frame

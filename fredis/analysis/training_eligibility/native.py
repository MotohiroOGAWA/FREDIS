"""Export independently verified FREDIS training-eligible SpecIDs."""
from __future__ import annotations
from collections import Counter
from concurrent.futures import ThreadPoolExecutor,as_completed
import hashlib
import json
import os
from pathlib import Path
import queue
import shutil
import subprocess
import sys
import threading
import torch
from .common import atomic_json,sha256,spec_ids,write_ids,write_tsv


class NativeWorker:
    def __init__(self,config_path,log_path):
        self.config_path=config_path;self.log=Path(log_path).open('a');self.child=None;self.reader=None;self.serial=0
    def close(self):
        if self.child is not None:
            if self.child.poll() is None:
                try:self.child.kill()
                except ProcessLookupError:pass
            self.child.wait();self.child.stdin.close()
            if self.reader:self.reader.join(timeout=2)
            self.child.stdout.close();self.child=None
    def call(self,task):
        if self.child is None or self.child.poll() is not None:
            self.close()
            self.child=subprocess.Popen([sys.executable,'-m','fredis.analysis.training_eligibility.worker',str(self.config_path)],
                stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=self.log,text=True,start_new_session=False,
                env={**os.environ,'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','PYTHONUNBUFFERED':'1'})
            self.replies=queue.Queue()
            def read_lines(stream,replies):
                try:
                    for line in stream:replies.put(line)
                except (OSError,ValueError):pass
                finally:replies.put(None)
            self.reader=threading.Thread(target=read_lines,args=(self.child.stdout,self.replies),daemon=True)
            self.reader.start()
        self.serial+=1;request_id=self.serial
        self.child.stdin.write(json.dumps({**task,'request_id':request_id})+'\n');self.child.stdin.flush()
        while True:
            reply=self.replies.get()
            if reply is None:self.close();raise RuntimeError('Native worker exited; see run/worker logs')
            try:result=json.loads(reply)
            except ValueError:self.log.write(reply);self.log.flush();continue
            if not isinstance(result,dict) or result.get('request_id')!=request_id:continue
            if not result.get('ok'):raise RuntimeError(result['error'])
            return result['records']



def code_hash():
    root=Path(__file__).resolve().parents[2];digest=hashlib.sha256()
    for file in sorted(root.rglob('*.py')):
        digest.update(str(file.relative_to(root)).encode());digest.update(file.read_bytes())
    return digest.hexdigest()


def default_model(chemistry):
    from fredis.ml.training.fragment_tree_training.training import build_arg_parser,training_model_config
    from fredis.domain.fragment import Fragmenter
    args=build_arg_parser().parse_args(['--train-dir','unused','--val-dir','unused','--output-dir','unused'])
    model=training_model_config(args)
    model['fragmenter_params']=chemistry['fragmenter_params'];model['mol_encoder_params']['symbols']=chemistry['symbols']
    model['adduct_type_strs']=[str(value) for value in Fragmenter.from_dict(chemistry['fragmenter_params']).adduct_types]
    return model


def extract(input_path,output_dir,chemistry,*,model_config=None,mapping=None,id_column='SpecID',workers=1,
            batch_size=8,seed=42,resume=False,keep_export=False,minimum_relative_intensity=0,
            max_unique_fragment_smiles=-1,max_reaction_combinations=-1,progress=None):
    from rdkit import rdBase
    import pandas as pd
    from fredis.libs.msentity.msentity.core.MSDataset import MSDataset
    from fredis.ml.data_preparation.fragment_tree.record_validation import inspect_records,DEFAULT_MAPPING
    from fredis.ml.data_preparation.fragment_tree.context import create_preparation_context,validate_limits
    from fredis.ml.specgen.spectrum_generator import create_spectrum_generator
    from fredis.ml.training.fragment_tree_training.pattern_models import initialize_imports
    from fredis.ml.training.fragment_tree_training.model import ReactionFragmentTreeTrainingModel
    if workers<1 or batch_size<1:raise ValueError('workers and batch size must be positive')
    if not 0<=minimum_relative_intensity<=1:raise ValueError('minimum relative intensity must be in [0,1]')
    validate_limits(max_unique_fragment_smiles,max_reaction_combinations)
    input_path=Path(input_path).resolve();output=Path(output_dir).resolve()
    context=create_preparation_context(chemistry);mapping={**DEFAULT_MAPPING,**(mapping or {})}
    dataset=MSDataset.load(str(input_path),load_peak_metadata=False);ids=spec_ids(dataset,id_column);metadata=dataset.metadata
    model_config=json.loads(json.dumps(model_config or default_model(chemistry)))
    model_config=model_config.get('params',model_config)
    model_config['fragmenter_params']=chemistry['fragmenter_params'];model_config.setdefault('mol_encoder_params',{})['symbols']=chemistry['symbols']
    model_config['adduct_type_strs']=list(context.adduct_type_strs)
    # Fail global model/configuration errors before classifying individual spectra.
    torch.manual_seed(seed)
    generator=create_spectrum_generator(model_config)
    model=ReactionFragmentTreeTrainingModel(generator.feature_model,spectrum_model=generator.spectrum_model)
    if model_config.get('fine_tuning',{}).get('pattern_imports'):initialize_imports(model,model_config)
    model_config.setdefault('reaction_model_params',{})['pattern_model_params']=generator.feature_model.reaction_instance_encoder.pattern_model_params
    model_config['reaction_model_params']['max_roles']=generator.feature_model.reaction_instance_encoder.roles.num_embeddings
    model_config['mol_encoder_params']['descriptors']=list(generator.mol_encoder.descriptors)
    del generator,model
    dependencies={item['path']:sha256(item['path']) for item in model_config.get('fine_tuning',{}).get('pattern_imports',[])}
    config=dict(model_config=model_config,mapping=mapping,limits=dict(max_unique_fragment_smiles=max_unique_fragment_smiles,max_reaction_combinations=max_reaction_combinations),minimum_relative_intensity=minimum_relative_intensity,
        batch_size=batch_size,device='cpu',seed=seed)
    fingerprint=dict(input_path=str(input_path),input_sha256=sha256(input_path),code_sha256=code_hash(),config=config,
        dependencies=dependencies,id_column=id_column,workers=workers,torch_version=torch.__version__,rdkit_version=rdBase.rdkitVersion,python_version=sys.version)
    manifest_path=output/'run/manifest.json'
    if output.exists():
        if not resume:raise FileExistsError(f'Output already exists; use --resume: {output}')
        if not manifest_path.exists() or json.loads(manifest_path.read_text())!=fingerprint:raise ValueError('Resume rejected: input, code, configuration, dependencies or environment changed')
        if (output/'stats.json').exists():
            previous=json.loads((output/'stats.json').read_text())
            if previous.get('status')=='completed' and (output/'trainable_spec_ids.txt').exists():
                if previous.get('trainable_ids_sha256')!=sha256(output/'trainable_spec_ids.txt'):
                    raise ValueError('Completed SpecID list was modified; choose a new output directory')
                return previous
    else:
        if resume:raise ValueError('No previous run to resume')
        (output/'run').mkdir(parents=True);atomic_json(manifest_path,fingerprint)
    atomic_json(output/'model.architecture.pft', dict(model_config={key:model_config[key] for key in ['mol_encoder_params','reaction_model_params','spectrum_model_params'] if key in model_config}))
    for name in ['input','groups','prepared']:(output/'run'/name).mkdir(exist_ok=True)
    config_path=output/'run/native_config.json';atomic_json(config_path,config)
    inspection=inspect_records(dataset,context.fragmenter,mapping,symbols=chemistry['symbols'])
    rows={item['index']:dict(index=item['index'],Status='excluded',Stage='preflight',Reason='; '.join(issue['reason'] for issue in item['issues']),ReasonCategory='InvalidMetadata') for item in inspection['invalidRecords']}
    eligible=[]
    import numpy as np
    for index in inspection['validIndexes']:
        peaks=dataset[index].peaks
        if not np.isfinite(peaks.mz).all() or np.any(peaks.mz<=0) or not np.isfinite(peaks.intensity).all() or np.any(peaks.intensity<0):
            rows[index]=dict(index=index,Status='excluded',Stage='preflight',Reason='Peak m/z must be positive and finite; intensities must be non-negative and finite',ReasonCategory='InvalidPeaks')
        else:eligible.append(index)
    grouped={}
    smiles=metadata[mapping['smilesColumn']].astype(str).tolist()
    for index in eligible:grouped.setdefault(smiles[index],[]).append(index)
    tasks=[]
    for number,(smile,indices) in enumerate(grouped.items()):
        key=f'{number:08d}';saved=output/'run/groups'/(key+'.json')
        if saved.exists():
            records=json.loads(saved.read_text())
            if sorted(row['index'] for row in records)!=sorted(indices):raise ValueError('Saved group result has mismatched record IDs')
            rows.update({row['index']:row for row in records});continue
        file=output/'run/input'/(key+'.msds')
        if not file.exists():dataset[indices].save(str(file))
        tasks.append(dict(key=key,input=str(file),smiles=smile,record_indices=indices,workdir=str(output/'run/prepared'/key)))
    local=threading.local();proxies=[];lock=threading.Lock();stop=threading.Event()
    def process(task):
        if stop.is_set():raise RuntimeError('Cancelled')
        if not hasattr(local,'worker'):
            with lock:
                local.worker=NativeWorker(config_path,output/'run'/f'worker-{len(proxies)}.log');proxies.append(local.worker)
        worker=local.worker
        try:records=worker.call(task)
        except Exception:
            records=[]
            for position,index in enumerate(task['record_indices']):
                if stop.is_set():raise RuntimeError('Cancelled')
                try:records.extend(worker.call({**task,'positions':[position]}))
                except Exception as error:records.append(dict(index=index,Status='excluded',Stage='native_worker',Reason=str(error),ReasonCategory=type(error).__name__))
        if stop.is_set():raise RuntimeError('Cancelled')
        atomic_json(output/'run/groups'/(task['key']+'.json'),records)
        return records
    executor=ThreadPoolExecutor(max_workers=workers)
    try:
        futures=[executor.submit(process,task) for task in tasks]
        for future in as_completed(futures):
            records=future.result();rows.update({row['index']:row for row in records})
            event=dict(completed=len(rows),total=len(ids),trainable=sum(row['Status']=='trainable' for row in rows.values()))
            if progress:progress(event)
    finally:
        stop.set()
        for proxy in proxies:proxy.close()
        executor.shutdown(wait=True,cancel_futures=True)
        for proxy in proxies:proxy.log.close()
    if len(rows)!=len(ids):raise RuntimeError('Some records have not been processed')
    if sha256(input_path)!=fingerprint['input_sha256']:raise ValueError('Input changed during the run; results not published')
    ordered=[dict(SpecID=ids[index],**rows[index]) for index in range(len(ids))]
    accepted=[row['index'] for row in ordered if row['Status']=='trainable']
    reasons=Counter();stages=Counter();categories=Counter()
    for item in inspection['invalidRecords']:
        for issue in item['issues']:reasons[issue['reason']]+=1
    for row in ordered:
        if row['Status']=='excluded':
            stages[row['Stage']]+=1;categories[row['ReasonCategory']]+=1
            if row['ReasonCategory']!='InvalidMetadata':reasons[row['Reason']]+=1
    def breakdown(column):
        if column not in metadata:return {}
        counts=Counter(metadata[column].fillna('(missing)').astype(str));passed=Counter(metadata.iloc[accepted][column].fillna('(missing)').astype(str))
        return {key:dict(input=value,trainable=passed[key]) for key,value in sorted(counts.items())}
    stats=dict(status='completed',tool='fredis',input_path=str(input_path),input_sha256=fingerprint['input_sha256'],
        total_records=len(ids),n_input=len(ids),trainable=len(accepted),excluded=len(ids)-len(accepted),
        trainable_percent=100*len(accepted)/len(ids) if ids else 0,native_config=config,column_mapping=mapping,id_column=id_column,
        funnel=dict(input=len(ids),preflight_pass=len(eligible),native_checked=len(eligible),trainable=len(accepted)),
        excluded_by_stage=dict(stages),exclusion_reasons=dict(reasons),reason_categories=dict(categories),
        unique_smiles=dict(input=len(set(smiles)),trainable=len({smiles[i] for i in accepted})),
        adducts=breakdown(mapping['adductTypeColumn']),instruments=breakdown('InstrumentType'),
        code_sha256=fingerprint['code_sha256'],torch_version=torch.__version__,rdkit_version=rdBase.rdkitVersion)
    write_tsv(output/'records.tsv',['SpecID','index','Status','Stage','Reason','ReasonCategory','TrainingAdduct','TrainingMainAdduct','TrainingCollisionEnergy','AssignmentScore','Loss','GradientsFinite'],ordered)
    text=[f'FREDIS trainable: {len(accepted)} / {len(ids)} ({stats["trainable_percent"]:.2f}%)',f'Excluded: {stats["excluded"]}',f'Funnel: {stats["funnel"]}',f'SMILES: {stats["unique_smiles"]}']
    text.extend(f'{stage}: {count}' for stage,count in sorted(stages.items()));text.extend(f'{reason}: {count}' for reason,count in sorted(reasons.items()))
    for key in ['adducts','instruments']:
        text.append(key+':');text.extend(f'  {name}: {counts}' for name,counts in stats[key].items())
    (output/'stats.txt').write_text('\n'.join(text)+'\n')
    write_ids(output/'trainable_spec_ids.txt',(ids[index] for index in accepted))
    stats['trainable_ids_sha256']=sha256(output/'trainable_spec_ids.txt')
    atomic_json(output/'stats.json',stats)
    if not keep_export:
        shutil.rmtree(output/'run/input');shutil.rmtree(output/'run/prepared')
    return stats

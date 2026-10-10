"""Shared CLI configuration loading with explicit option precedence."""
from __future__ import annotations
import argparse
import copy
import json
from pathlib import Path

PRESET = Path(__file__).resolve().parents[2] / 'presets/spectrum_generator_params/source_anchored_pos_model_config.json'
OPTIONS = {
    'max_reaction_count': ('fragmenter_params', 'fragment_ion_tree_builder', 'max_reaction_count'),
    'precursor_max_reaction_count': ('fragmenter_params', 'precursor_candidate_max_reaction_count'),
    'mass_tolerance': ('fragmenter_params', 'mass_tolerance'),
    'branch_path_threshold': ('generation_params', 'branch_path_threshold'),
    'max_fragment_nodes': ('generation_params', 'max_fragment_nodes'),
}

def configure_model_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument('--params', help='Model or fragmenter JSON file. Explicit options override file values.')
    parser.add_argument('--params-json', help='Inline model or fragmenter JSON; overrides --params.')
    for group in ('fragmenter', 'mol-encoder', 'reaction-model', 'spectrum-model', 'generation'):
        parser.add_argument('--'+group+'-params-json', help='Inline parameter object overriding the corresponding configuration section.')
    for key in OPTIONS:
        kind = str if key == 'mass_tolerance' else float if key == 'branch_path_threshold' else int
        parser.add_argument('--'+key.replace('_','-'), type=kind, help='Override '+'.'.join(OPTIONS[key])+'.')
    parser.add_argument('--set', dest='parameter_overrides', action='append', default=[], metavar='PATH=JSON',
                        help='Override any individual parameter, including nested array entries. Repeat as needed; applied last.')

def merge(base: dict, update: dict) -> dict:
    result=copy.deepcopy(base)
    for key,value in update.items():
        result[key]=merge(result[key],value) if isinstance(value,dict) and isinstance(result.get(key),dict) else copy.deepcopy(value)
    return result

def unpack(config: dict) -> dict:
    config=config.get('params',config)
    if not isinstance(config,dict): raise ValueError('Configuration must be a JSON object')
    # Analysis and Workbench result/config files wrap the same model or
    # Fragmenter payload under these explicit keys. Accepting them here keeps
    # every CLI using one precedence/override implementation.
    config=config.get('model_config',config.get('modelConfig',config))
    if 'fragmenterParams' in config:
        config={'fragmenter_params':config['fragmenterParams'],**({'mol_encoder_params':{'symbols':config['symbols']}} if config.get('symbols') else {})}
    if 'fragment_ion_tree_builder' in config: return {'fragmenter_params':config}
    return config

def set_parameter(config: dict, path: list[str], value: object) -> None:
    if not path or any(key in ('__proto__','prototype','constructor') or not key for key in path):
        raise ValueError('Invalid parameter path')
    parent=config
    for key in path[:-1]:
        if isinstance(parent,list): parent=parent[int(key)]
        else:
            if key not in parent: parent[key]={}
            parent=parent[key]
    if isinstance(parent,list): parent[int(path[-1])]=value
    else: parent[path[-1]]=value

def resolve_model_options(args: argparse.Namespace) -> dict:
    config=json.loads(PRESET.read_text())
    if args.params: config=merge(config,unpack(json.loads(Path(args.params).read_text())))
    if args.params_json: config=merge(config,unpack(json.loads(args.params_json)))
    for group in ('fragmenter','mol_encoder','reaction_model','spectrum_model','generation'):
        raw=getattr(args,group+'_params_json',None)
        if raw:
            value=json.loads(raw)
            if not isinstance(value,dict): raise ValueError(group+' parameters must be a JSON object')
            config[group+'_params']=merge(config.get(group+'_params',{}),value)
    for key,path in OPTIONS.items():
        value=getattr(args,key,None)
        if value is not None: set_parameter(config,list(path),value)
    for override in args.parameter_overrides:
        key,separator,raw=override.partition('=')
        if not separator: raise ValueError('--set requires PATH=JSON')
        set_parameter(config,key.split('.'),json.loads(raw))
    # Supported adducts are derived from Fragmenter rules and observed dataset
    # metadata during preparation, never from a user configuration override.
    config.pop('adduct_type_strs', None)
    return config

def namespace_argv(parser: argparse.ArgumentParser, args: argparse.Namespace) -> list[str]:
    argv=[]
    for action in parser._actions:
        if isinstance(action,argparse._HelpAction) or not action.option_strings: continue
        value=getattr(args,action.dest,None)
        if value is None: continue
        if isinstance(action,argparse._StoreTrueAction):
            if value:argv.append(action.option_strings[0])
        elif isinstance(action,argparse._StoreFalseAction):
            if not value:argv.append(action.option_strings[0])
        elif isinstance(action,argparse._AppendAction):
            for item in value: argv.extend([action.option_strings[0],str(item)])
        elif action.nargs in ('*','+'):
            argv.extend([action.option_strings[0],*(str(item) for item in value)])
        else: argv.extend([action.option_strings[0],str(value)])
    return argv

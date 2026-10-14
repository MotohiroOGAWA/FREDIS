from __future__ import annotations
import csv
import hashlib
import json
from pathlib import Path
import pandas as pd


def sha256(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):digest.update(block)
    return digest.hexdigest()


def atomic_json(path,value):
    path=Path(path);temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n');temporary.replace(path)


def spec_ids(dataset,column='SpecID'):
    if column not in dataset.columns:raise ValueError(f'Missing SpecID column: {column}')
    values=dataset.metadata[column].tolist();result=[];seen=set()
    for index,value in enumerate(values):
        if pd.isna(value):raise ValueError(f'Missing SpecID at input row {index}')
        text=str(value)
        if not text.strip() or text!=text.strip() or '\n' in text or '\r' in text or '\t' in text:
            raise ValueError(f'SpecID at row {index} must be nonempty, single-line and have no surrounding whitespace')
        if text in seen:raise ValueError(f'Duplicate SpecID: {text}')
        seen.add(text);result.append(text)
    return result


def write_ids(path,ids):
    path=Path(path);temporary=path.with_suffix(path.suffix+'.tmp')
    with temporary.open('w') as stream:
        for value in ids:stream.write(value+'\n')
    temporary.replace(path)


def write_tsv(path,fields,rows):
    with Path(path).open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields,delimiter='\t',extrasaction='ignore')
        writer.writeheader();writer.writerows(rows)


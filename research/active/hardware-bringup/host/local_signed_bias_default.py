"""Resolve the explicit simulation Development default, with contract hashes."""
from pathlib import Path
import hashlib,json

ROOT=Path(__file__).resolve().parents[4]
POINTER=ROOT/'research/active/dtr-r0/nearfield/CNH_SIM_DEFAULT_CANDIDATE.json'

def default_manifest():
    pointer=json.loads(POINTER.read_text(encoding='utf8'))
    manifest=ROOT/pointer['manifest_path']
    digest=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    if digest(manifest)!=pointer['manifest_sha256']:
        raise ValueError('Simulation default manifest hash mismatch')
    contract=ROOT/pointer['contract_path']
    if digest(contract)!=pointer['contract_sha256']:
        raise ValueError('Simulation default contract hash mismatch')
    return manifest

def default_engine(**kwargs):
    from local_signed_bias_adapter import ContractReplay
    return ContractReplay(default_manifest(),**kwargs)

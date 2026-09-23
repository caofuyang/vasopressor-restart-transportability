from pathlib import Path
import hashlib
import json
import os
import pandas as pd

CODE=Path(__file__).resolve().parent
HERE=Path(os.environ['EICU_RUN']); HERE.mkdir(parents=True,exist_ok=True)
RAW=Path(os.environ['EICU_AUDIT_RAW'])
DERIVED=Path(os.environ['EICU_AUDIT_DERIVED'])
PARAMETERS=Path(os.environ['REPRO_PARAMETERS'])
ORIGINAL_SOURCE=CODE.parent/'eicu'
PARTITION=CODE/'eicu_target_partition_freeze_manifest.json'
RESULTS=HERE/'results'
PRIVATE=HERE/'private'
RULE_FILE=CODE/'00_EICU_CORRECTION_RULES_FROZEN.md'
LOCK_FILE=CODE/'RULES_LOCK.sha256'
RULE_SHA=LOCK_FILE.read_text(encoding='utf-8').split()[0]
VARIANTS={'primary_traceable':(None,0),'freshness_180':(180,0),'gap_60':(None,60)}
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''): h.update(b)
    return h.hexdigest()
def check_rules():
    assert sha(RULE_FILE)==RULE_SHA,'Frozen rules changed'
def write_json(p,obj):
    with Path(p).open('x',encoding='utf-8') as f: json.dump(obj,f,ensure_ascii=False,indent=2,default=str)
def csv(name,df):
    p=RESULTS/name
    assert not p.exists(),str(p)
    df.to_csv(p,index=False,encoding='utf-8-sig')
def inputs():
    return [DERIVED/'stage1b_explicit_stop_cohort.parquet',DERIVED/'stage2b_explicit_stop_feature_matrix.parquet',
      RAW/'patient.csv.gz',RAW/'infusionDrug.csv.gz',RAW/'carePlanEOL.csv.gz',
      ORIGINAL_SOURCE/'eicu_stage1b_explicit_stop_cohort.py',ORIGINAL_SOURCE/'eicu_stage2_external_features.py',PARTITION,
      *sorted(PARAMETERS.glob('*.json'))]
def snapshot(): return {str(p):{'sha256':sha(p),'bytes':p.stat().st_size} for p in inputs()}

def union_intervals(intervals):
    merged=[]
    for s,e in sorted(intervals):
        if pd.isna(s) or pd.isna(e) or e<s: continue
        if not merged or s>merged[-1][1]: merged.append([float(s),float(e)])
        else: merged[-1][1]=max(merged[-1][1],float(e))
    return merged

def fresh_candidate(events,t,cap):
    """Finite state expiry is UNKNOWN; fixed origin only, no replacement candidate.
    events: list of (time,drug,rate) with valid source-unit times, deduplicated.
    Returns reason. A never-charted agent is not assigned a zero state.
    """
    bydrug={}
    for minute,drug,rate in events:
        if minute<=t: bydrug.setdefault(drug,[]).append((minute,rate))
    if not bydrug: return 'indeterminate_freshness_no_states'
    positive=[]
    for rows in bydrug.values():
        rows.sort()
        last,rate=rows[-1]
        if t-last>cap: return 'indeterminate_freshness_expired_state_at_origin'
        if rate!=0: return 'invalid_freshness_nonzero_at_origin'
        for i,(minute,rate) in enumerate(rows):
            end=min(t,minute+cap,rows[i+1][0] if i+1<len(rows) else t)
            if rate>0 and end>minute: positive.append((minute,end))
    # An expired positive state cannot bridge an unknown gap into a 6h episode.
    if not any(s<=t-360 and e>=t for s,e in union_intervals(positive)):
        return 'indeterminate_freshness_six_hour_exposure_not_supported'
    return None

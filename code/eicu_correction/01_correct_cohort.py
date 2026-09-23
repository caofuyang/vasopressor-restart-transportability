"""Post-audit eICU correction. Writes only independent private data and aggregates."""
import ast
from datetime import datetime,timezone
import json
import math
import duckdb
import numpy as np
import pandas as pd
from common import *

check_rules()
RESULTS.mkdir(exist_ok=False)
PRIVATE.mkdir(exist_ok=False)
write_json(HERE/'RULES_LOCK.json',{'utc_before_new_performance':datetime.now(timezone.utc).isoformat(),'sha256':RULE_SHA,'rules':'00_RULES_FROZEN.md','variants':VARIANTS})
before=snapshot()
write_json(RESULTS/'input_snapshot_before.json',before)
con=duckdb.connect(':memory:')
con.execute("SET threads=1; SET memory_limit='6GB'; SET temp_directory=''")
def qpath(p): return str(p).replace('\\','/').replace("'","''")
for name,file,varchar in [('patient','patient.csv.gz',False),('infusion','infusionDrug.csv.gz',True),('eol','carePlanEOL.csv.gz',False)]:
    con.execute(f"CREATE VIEW {name} AS SELECT * FROM read_csv_auto('{qpath(RAW/file)}',all_varchar={str(varchar).lower()})")
con.execute(f"CREATE VIEW f AS SELECT * FROM read_parquet('{qpath(DERIVED/'stage2b_explicit_stop_feature_matrix.parquet')}')")
allowed={'events','timeline','transitions','episodes','patient_one','candidates','eol_flags','final'}
for node in ast.walk(ast.parse((ORIGINAL_SOURCE/'eicu_stage1b_explicit_stop_cohort.py').read_text(encoding='utf-8-sig'))):
    if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute) and node.func.attr=='execute' and node.args and isinstance(node.args[0],ast.Constant) and isinstance(node.args[0].value,str):
        query=node.args[0].value
        tok=query.split()
        if len(tok)>3 and [x.upper() for x in tok[:3]]==['CREATE','TEMP','TABLE'] and tok[3] in allowed: con.execute(query)
con.execute("""CREATE TEMP TABLE units AS SELECT *, -hospitaladmitoffset unit_start,
unitdischargeoffset-hospitaladmitoffset unit_end,hospitaldischargeoffset-hospitaladmitoffset hosp_end
FROM patient WHERE patienthealthsystemstayid IN (SELECT patienthealthsystemstayid FROM f)""")
con.execute("""CREATE TEMP TABLE records AS WITH chosen AS (
SELECT patienthealthsystemstayid,hospital_minute,drug,arg_max(rate,infusiondrugid) rate,max(infusiondrugid) iid
FROM events WHERE drug IS NOT NULL AND rate IS NOT NULL AND rate>=0
AND patienthealthsystemstayid IN(SELECT patienthealthsystemstayid FROM f) GROUP BY ALL)
SELECT c.*,u.patientunitstayid,u.uniquepid,
(c.hospital_minute>=u.unit_start AND c.hospital_minute<=u.unit_end AND u.patienthealthsystemstayid=c.patienthealthsystemstayid) valid_time
FROM chosen c JOIN infusion i ON try_cast(i.infusiondrugid AS BIGINT)=c.iid
JOIN units u ON u.patientunitstayid=try_cast(i.patientunitstayid AS BIGINT)""")
f=con.execute('SELECT * FROM f ORDER BY patienthealthsystemstayid').df()
unitgroups={int(k):v for k,v in con.execute('SELECT * FROM units').df().groupby('patienthealthsystemstayid')}
recordgroups={int(k):v.sort_values(['hospital_minute','drug']) for k,v in con.execute('SELECT * FROM records').df().groupby('patienthealthsystemstayid')}
partition=json.loads(PARTITION.read_text(encoding='utf-8'))['partitions']
reverse={int(h):p for p,hs in partition.items() for h in hs}
assert len(reverse)==sum(map(len,partition.values()))
cohortcols=[r[0] for r in con.execute('DESCRIBE final').fetchall()]
checks=' OR '.join(f'a."{c}" IS DISTINCT FROM b."{c}"' for c in cohortcols)
mismatch=con.execute(f'SELECT count(*) FROM final a FULL JOIN f b USING(patienthealthsystemstayid) WHERE {checks}').fetchone()[0]
assert mismatch==0
assert len(f)==8810 and f.patienthealthsystemstayid.nunique()==8810
rows=[]
omitted_detail=[]
for r in f.itertuples(index=False):
    sid=int(r.patienthealthsystemstayid); t=float(r.episode_end_minute); h=float(r.hospital_discharge_minute)
    u=unitgroups[sid]; rec=recordgroups[sid]
    intervals=union_intervals(list(zip(u.unit_start,u.unit_end)))
    unit_end=next((e for s,e in intervals if s<=t<=e),np.nan)
    obs_end=min(unit_end,h) if np.isfinite(unit_end) and np.isfinite(h) else np.nan
    hs=u.hospitaldischargestatus.fillna('').str.lower()
    known=hs[hs.isin(['alive','expired'])]
    missing_status=not hs.isin(['alive','expired']).all()
    hosp_deaths=sorted(u.loc[hs=='expired','hosp_end'].dropna().unique())
    unit_deaths=sorted(u.loc[u.unitdischargestatus.fillna('').str.lower()=='expired','unit_end'].dropna().unique())
    proxies=list(hosp_deaths)+list(unit_deaths)
    death=float(hosp_deaths[0]) if len(hosp_deaths)==1 else np.nan
    proxy_le=any(d<=t for d in proxies)
    death_position='before' if any(d<t for d in proxies) else ('equal' if any(d==t for d in proxies) else 'after_or_unavailable')
    conflict=(known.nunique()>1 or u.hosp_end.nunique()!=1 or len(unit_deaths)>1 or len(hosp_deaths)>1
        or bool(unit_deaths) and (len(hosp_deaths)!=1 or any(d!=hosp_deaths[0] for d in unit_deaths)))
    identity=(u.uniquepid.nunique()!=1 or str(u.uniquepid.iloc[0])!=str(r.uniquepid) or u.hospitalid.nunique()!=1 or not np.isfinite(t) or not np.isfinite(h))
    positive=rec[(rec.rate>0)&(rec.hospital_minute>t)&(rec.hospital_minute<=t+1440)]
    valid=positive[positive.valid_time.fillna(False)&(positive.uniquepid.astype(str)==str(r.uniquepid))]
    invalid=positive[~(positive.valid_time.fillna(False)&(positive.uniquepid.astype(str)==str(r.uniquepid)))]
    restart=float(valid.hospital_minute.min()) if len(valid) else np.nan
    first_raw=float(positive.hospital_minute.min()) if len(positive) else np.nan
    bad_restart=float(invalid.hospital_minute.min()) if len(invalid) else np.nan
    death24=death if np.isfinite(death) and t<death<=t+1440 else np.nan
    first=min([v for v in [restart,death24] if np.isfinite(v)],default=t+1440)
    omitted=(r.hospital_discharge_minute>=t and r.restart_24h==0 and np.isfinite(first_raw))
    earlier_restart=(np.isfinite(restart) and (not np.isfinite(death24) or restart<death24))
    base=dict(patienthealthsystemstayid=sid,partition=reverse[int(r.hospitalid)],old_outcome=r.competing_outcome,
       old_code=int(r.competing_outcome_code),origin=t,restart=restart,death_proxy=death,observation_end=obs_end,
       omitted_restart_evidence=omitted,proxy_position=death_position,flag_identity=bool(identity),flag_origin=not np.isfinite(unit_end),
       flag_proxy_le=bool(proxy_le),flag_missing_status=bool(missing_status),flag_proxy_conflict=bool(conflict))
    valid_history=[(float(x.hospital_minute),x.drug,float(x.rate)) for x in rec[rec.valid_time.fillna(False)].itertuples(index=False)]
    for variant,(cap,gap) in VARIANTS.items():
        status=None; code=-1
        if identity: status='invalid_identity_or_index'
        elif missing_status: status='indeterminate_missing_hospital_status'
        elif not np.isfinite(unit_end): status='invalid_origin_outside_unit'
        elif proxy_le: status='invalid_death_proxy_'+death_position+'_origin'
        elif conflict: status='indeterminate_death_or_discharge_proxy_conflict'
        elif (known=='alive').all() and h<t: status='invalid_alive_discharge_before_origin'
        if status is None and cap is not None: status=fresh_candidate(valid_history,t,cap)
        if status is None and gap:
            if np.isfinite(first_raw) and first_raw<=t+gap: status='invalid_minimum_stop_interval'
            elif obs_end<t+gap or np.isfinite(death24) and death24<=t+gap: status='indeterminate_minimum_stop_interval'
        if status is None:
            if np.isfinite(bad_restart) and bad_restart<=first: status='invalid_or_indeterminate_restart_source'
            elif np.isfinite(restart) and np.isfinite(death24) and restart==death24: status='indeterminate_simultaneous_events'
            elif not np.isfinite(obs_end) or obs_end<first: status='indeterminate_observation'
            elif earlier_restart: status='recorded_restart_first'; code=1
            elif np.isfinite(death24): status='recorded_death_proxy_first'; code=2
            else: status='no_recorded_restart_or_death_proxy_under_observation'; code=0
        rows.append(dict(base,variant=variant,status=status,new_code=code,included=code>=0))
    if omitted:
        # Each record has raw evidence, even if eligibility/observation later excludes it.
        assert len(positive)>0
        omitted_detail.append(dict(base,raw_first_positive=first_raw,positive_evidence_rows=len(positive),
            valid_positive_evidence_rows=len(valid),closed_episode_at_first=bool(con.execute('SELECT count(*) FROM episodes WHERE patienthealthsystemstayid=? AND episode_start_minute=?',[sid,first_raw]).fetchone()[0])))
labels=pd.DataFrame(rows)
omitted=pd.DataFrame(omitted_detail)
assert len(omitted)==738 and omitted.closed_episode_at_first.sum()==0
primary=labels[labels.variant=='primary_traceable']
assert primary.patienthealthsystemstayid.nunique()==8810
labels.to_parquet(PRIVATE/'corrected_labels.parquet',index=False)
omitted.to_parquet(PRIVATE/'omitted_738_record_checks.parquet',index=False)
csv('exclusion_flow.csv',labels.groupby(['variant','partition','status','included'],dropna=False).size().reset_index(name='n'))
csv('label_transition.csv',labels.groupby(['variant','partition','old_outcome','status','new_code'],dropna=False).size().reset_index(name='n'))
csv('omitted_738_disposition.csv',primary[primary.omitted_restart_evidence].groupby(['partition','old_outcome','status']).size().reset_index(name='n'))
flags=[c for c in labels if c.startswith('flag_')]
csv('nonexclusive_flags.csv',primary.groupby('partition')[flags].sum().reset_index())
csv('cohort_counts.csv',labels.groupby(['variant','partition']).agg(n=('included','size'),included=('included','sum')).reset_index())
csv('death_origin_position.csv',primary[primary.flag_proxy_le].groupby(['partition','proxy_position','status']).size().reset_index(name='n'))
# Fixed-index feature reuse must be exact. No newly shifted origin is allowed.
check=labels.merge(f[['patienthealthsystemstayid','episode_end_minute']],on='patienthealthsystemstayid',validate='many_to_one')
assert (check.origin==check.episode_end_minute).all()
assert (primary[primary.included].restart.dropna()>primary[primary.included].origin.loc[primary[primary.included].restart.dropna().index]).all()
write_json(RESULTS/'cohort_QA.json',{'source_replay_mismatches':int(mismatch),'n_original':len(f),'omitted_records_checked':len(omitted),
 'omitted_closed_episode_count':int(omitted.closed_episode_at_first.sum()),'all_origins_unchanged':True,'no_feature_reanchoring':True,
 'partitions_unchanged':True,'rules_sha256':RULE_SHA,'patient_level_outputs':'private only; excluded from package'})
assert snapshot()==before
print(labels.groupby(['variant','included']).size().to_string(),flush=True)
print('COHORT COMPLETED, NO PERFORMANCE CALCULATED',flush=True)

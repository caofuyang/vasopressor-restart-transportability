from pathlib import Path
import os

import duckdb
import sys
import pyarrow as pa
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"mimic"))
from stream_raw import extract


ROOT = Path(os.environ["EICU_RAW"])
DERIVED = Path(os.environ["EICU_DERIVED"])
COHORT = DERIVED / "stage1b_explicit_stop_cohort.parquet"
OUTPUT = DERIVED / "stage2b_explicit_stop_feature_matrix.parquet"
SUMMARY = DERIVED / "stage2b_explicit_stop_feature_coverage.csv"


def p(path: Path) -> str:
    return str(path).replace("\\", "/").replace("'", "''")


con = duckdb.connect()
con.execute("PRAGMA threads=1")
con.execute("PRAGMA memory_limit='8GB'")
con.execute("SET temp_directory='" + os.environ["REPRO_TEMP"].replace("\\", "/").replace("'", "''") + "'")
con.execute(f"CREATE VIEW cohort AS SELECT * FROM read_parquet('{p(COHORT)}')")
con.execute(f"CREATE VIEW patient AS SELECT * FROM read_csv_auto('{p(ROOT/'patient.csv.gz')}', header=true)")
con.execute(
    f"CREATE VIEW infusion AS SELECT ordinality AS source_row_number, * EXCLUDE(ordinality) FROM read_csv_auto('{p(ROOT/'infusionDrug.csv.gz')}', "
    "header=true, all_varchar=true, parallel=false) WITH ORDINALITY"
)
# Vitals views are created after the bounded raw scan below.
# Vitals views are created after the bounded raw scan below.

# One row per ICU stay supplies the hospital-relative clock and a defensible weight.
con.execute(
    r"""
    CREATE TEMP TABLE patient_units AS
    SELECT
        patientunitstayid,
        patienthealthsystemstayid,
        hospitaladmitoffset,
        hospitalid,
        CASE
            WHEN admissionweight BETWEEN 30 AND 300 THEN admissionweight
            WHEN dischargeweight BETWEEN 30 AND 300 THEN dischargeweight
            ELSE NULL
        END fallback_weight,
        unitadmitsource,
        hospitaladmitsource
    FROM patient
    WHERE patienthealthsystemstayid IN (SELECT patienthealthsystemstayid FROM cohort)
    """
)

# Preserve original CSV record ordinal using bounded batches before filtering.
unit_keys = con.execute("SELECT patientunitstayid FROM patient_units").fetchdf().patientunitstayid.tolist()
periodic_raw = Path(os.environ["REPRO_TEMP"]) / 'eicu_periodic_filtered.parquet'
aperiodic_raw = Path(os.environ["REPRO_TEMP"]) / 'eicu_aperiodic_filtered.parquet'
periodic_types = {'patientunitstayid':pa.int64(),'observationoffset':pa.int64(),'heartrate':pa.float64(),'respiration':pa.float64(),'sao2':pa.float64(),'temperature':pa.float64(),'systemicmean':pa.float64()}
aperiodic_types = {'patientunitstayid':pa.int64(),'observationoffset':pa.int64(),'noninvasivemean':pa.float64()}
extract(ROOT/'vitalPeriodic.csv.gz',periodic_raw,periodic_types,key='patientunitstayid',keys=unit_keys)
extract(ROOT/'vitalAperiodic.csv.gz',aperiodic_raw,aperiodic_types,key='patientunitstayid',keys=unit_keys)
con.execute(f"CREATE VIEW vp AS SELECT * FROM read_parquet('{p(periodic_raw)}')")
con.execute(f"CREATE VIEW va AS SELECT * FROM read_parquet('{p(aperiodic_raw)}')")

# Classify every vaso chart point. An unresolved or ml/hour state remains NULL;
# no concentration is inferred. This prevents optimistic dose harmonization.
con.execute(
    r"""
    CREATE TEMP TABLE vaso AS
    SELECT
        u.patienthealthsystemstayid, i.source_row_number,
        try_cast(i.infusionoffset AS BIGINT)-u.hospitaladmitoffset hospital_minute,
        CASE
            WHEN regexp_matches(lower(i.drugname), 'angiotensin') THEN 'angiotensin_ii'
            WHEN regexp_matches(lower(i.drugname), 'norepinephrine|noradrenaline|levophed') THEN 'norepinephrine'
            WHEN regexp_matches(lower(i.drugname), 'phenylephrine|neo[- ]?synephrine') THEN 'phenylephrine'
            WHEN regexp_matches(lower(i.drugname), 'vasopressin') THEN 'vasopressin'
            WHEN regexp_matches(lower(i.drugname), 'epinephrine|adrenaline')
                 AND NOT regexp_matches(lower(i.drugname), 'norepinephrine|noradrenaline') THEN 'epinephrine'
            WHEN regexp_matches(lower(i.drugname), 'dopamine') THEN 'dopamine'
        END drug,
        try_cast(regexp_extract(coalesce(i.drugrate,''), '[-+]?[0-9]*\.?[0-9]+', 0) AS DOUBLE) rate,
        CASE
            WHEN try_cast(i.patientweight AS DOUBLE) BETWEEN 30 AND 300 THEN try_cast(i.patientweight AS DOUBLE)
            ELSE u.fallback_weight
        END weight_kg,
        CASE
            WHEN regexp_matches(lower(i.drugname), 'mcg/kg/min|ug/kg/min') THEN 'mcg/kg/min'
            WHEN regexp_matches(lower(i.drugname), 'mcg/min|ug/min') THEN 'mcg/min'
            WHEN regexp_matches(lower(i.drugname), 'ng/kg/min') THEN 'ng/kg/min'
            WHEN regexp_matches(lower(i.drugname), 'units/min|unit/min') THEN 'units/min'
            WHEN regexp_matches(lower(i.drugname), 'units/hr|units/hour|unit/hr|unit/hour') THEN 'units/hour'
            WHEN regexp_matches(lower(i.drugname), 'mg/kg/min') THEN 'mg/kg/min'
            WHEN regexp_matches(lower(i.drugname), 'ml/hr') THEN 'ml/hour'
            ELSE 'unit_unresolved'
        END dose_unit
    FROM infusion i
    INNER JOIN patient_units u
      ON try_cast(i.patientunitstayid AS BIGINT)=u.patientunitstayid
    WHERE i.drugname IS NOT NULL
      AND try_cast(i.infusionoffset AS BIGINT) IS NOT NULL
      AND regexp_matches(lower(i.drugname),
          'norepinephrine|noradrenaline|levophed|phenylephrine|neo[- ]?synephrine|vasopressin|epinephrine|adrenaline|dopamine|angiotensin')
    """
)
con.execute(
    r"""
    CREATE TEMP TABLE vaso_nee AS
    SELECT *,
        CASE
            WHEN rate IS NULL OR rate<=0 THEN NULL
            WHEN drug='norepinephrine' AND dose_unit='mcg/kg/min' THEN rate
            WHEN drug='norepinephrine' AND dose_unit='mcg/min' AND weight_kg IS NOT NULL THEN rate/weight_kg
            WHEN drug='norepinephrine' AND dose_unit='mg/kg/min' THEN rate*1000
            WHEN drug='epinephrine' AND dose_unit='mcg/kg/min' THEN rate
            WHEN drug='epinephrine' AND dose_unit='mcg/min' AND weight_kg IS NOT NULL THEN rate/weight_kg
            WHEN drug='epinephrine' AND dose_unit='mg/kg/min' THEN rate*1000
            WHEN drug='phenylephrine' AND dose_unit='mcg/kg/min' THEN rate*0.1
            WHEN drug='phenylephrine' AND dose_unit='mcg/min' AND weight_kg IS NOT NULL THEN rate/weight_kg*0.1
            WHEN drug='dopamine' AND dose_unit='mcg/kg/min' THEN rate*0.01
            WHEN drug='dopamine' AND dose_unit='mcg/min' AND weight_kg IS NOT NULL THEN rate/weight_kg*0.01
            WHEN drug='vasopressin' AND dose_unit='units/min' THEN rate*2.5
            WHEN drug='vasopressin' AND dose_unit='units/hour' THEN rate/24.0
            WHEN drug='angiotensin_ii' AND dose_unit='ng/kg/min' THEN rate/100.0
            WHEN drug='angiotensin_ii' AND dose_unit='mcg/kg/min' THEN rate*10.0
            ELSE NULL
        END nee
    FROM vaso
    WHERE drug IS NOT NULL
    """
)

# Episode descriptors and terminal drug. Tied terminal agents are resolved by the
# largest safely converted NEE; ties without a dose use a stable lexical order.
con.execute(
    r"""
    CREATE TEMP TABLE episode_descriptors AS
    WITH within_episode AS (
        SELECT c.patienthealthsystemstayid, c.episode_start_minute, c.episode_end_minute,
               v.hospital_minute, v.drug, v.nee
        FROM cohort c JOIN vaso_nee v USING(patienthealthsystemstayid)
        WHERE v.hospital_minute BETWEEN c.episode_start_minute AND c.episode_end_minute
    ), last_time AS (
        SELECT patienthealthsystemstayid, max(hospital_minute) terminal_minute
        FROM within_episode GROUP BY patienthealthsystemstayid
    ), ranked_terminal AS (
        SELECT w.*,
               row_number() OVER (
                   PARTITION BY w.patienthealthsystemstayid
                   ORDER BY coalesce(w.nee,-1) DESC, w.drug
               ) rn
        FROM within_episode w JOIN last_time l USING(patienthealthsystemstayid)
        WHERE w.hospital_minute=l.terminal_minute
    ), distinct_counts AS (
        SELECT patienthealthsystemstayid, count(DISTINCT drug) distinct_drugs
        FROM within_episode GROUP BY patienthealthsystemstayid
    )
    SELECT d.patienthealthsystemstayid, d.distinct_drugs
    FROM distinct_counts d JOIN ranked_terminal r USING(patienthealthsystemstayid)
    WHERE r.rn=1
    """
)

# Construct five-minute dose states using last observation carried forward for at
# most 180 minutes. If any currently charted agent has an unresolved dose, the NEE
# at that grid point is missing rather than partially summed.
con.execute(
    r"""
    CREATE TEMP TABLE dose_grid AS
    SELECT c.patienthealthsystemstayid, g.grid_minute
    FROM cohort c,
         unnest(range(c.episode_end_minute-360, c.episode_end_minute+1, 5)) g(grid_minute)
    """
)
con.execute(
    r"""
    CREATE TEMP TABLE latest_drug_states AS
    WITH candidates AS (
        SELECT g.patienthealthsystemstayid, g.grid_minute, v.drug, v.nee, v.dose_unit,
               row_number() OVER (
                   PARTITION BY g.patienthealthsystemstayid,g.grid_minute,v.drug
                   ORDER BY v.hospital_minute DESC, v.source_row_number ASC
               ) rn
        FROM dose_grid g JOIN vaso_nee v
          ON v.patienthealthsystemstayid=g.patienthealthsystemstayid
         AND v.hospital_minute BETWEEN g.grid_minute-180 AND g.grid_minute
    )
    SELECT * EXCLUDE(rn) FROM candidates WHERE rn=1
    """
)
con.execute(
    r"""
    CREATE TEMP TABLE dose_features AS
    WITH summed AS (
        SELECT patienthealthsystemstayid, grid_minute,
               count(*) active_agents,
               sum(CASE WHEN nee IS NULL THEN 1 ELSE 0 END) unresolved_agents,
               CASE WHEN sum(CASE WHEN nee IS NULL THEN 1 ELSE 0 END)=0
                    THEN sum(nee) ELSE NULL END total_nee
        FROM latest_drug_states
        GROUP BY patienthealthsystemstayid, grid_minute
    ), anchored AS (
        SELECT s.*, c.episode_end_minute, c.episode_start_minute
        FROM summed s JOIN cohort c USING(patienthealthsystemstayid)
    )
    SELECT patienthealthsystemstayid,
           arg_max(total_nee,grid_minute) FILTER (WHERE total_nee IS NOT NULL) nee_last_5min,
           arg_max(active_agents,grid_minute) FILTER (WHERE total_nee IS NOT NULL) active_agents_last_5min,
           avg(total_nee) FILTER (WHERE grid_minute>episode_end_minute-60) nee_mean_1h,
           max(total_nee) FILTER (WHERE grid_minute>episode_end_minute-60) nee_max_1h,
           avg(total_nee) nee_mean_6h,
           max(total_nee) nee_max_6h,
           regr_slope(total_nee,grid_minute) FILTER (WHERE total_nee IS NOT NULL) nee_slope_toward_stop_6h,
           max(active_agents) FILTER (WHERE total_nee IS NOT NULL) maximum_active_agents_6h,
           100.0*avg(CASE WHEN total_nee=0 THEN 1.0 ELSE 0.0 END) FILTER (WHERE total_nee IS NOT NULL) zero_nee_grid_pct_6h,
           count(total_nee) dose_grid_observed,
           count(*) dose_grid_present,
           sum(CASE WHEN unresolved_agents>0 THEN 1 ELSE 0 END) dose_grid_unresolved
    FROM anchored GROUP BY patienthealthsystemstayid
    """
)

# Periodic vitals are typically five-minute values. Values outside physiologic
# ranges are removed before summarization. Arterial MAP is preferred at a shared
# minute; non-invasive MAP fills minutes without a valid arterial value.
con.execute(
    r"""
    CREATE TEMP TABLE periodic_window AS
    SELECT c.patienthealthsystemstayid, c.episode_end_minute, v.source_row_number,
           try_cast(v.observationoffset AS BIGINT)-u.hospitaladmitoffset hospital_minute,
           CASE WHEN v.heartrate BETWEEN 20 AND 250 THEN v.heartrate END heart_rate,
           CASE WHEN v.respiration BETWEEN 4 AND 80 THEN v.respiration END respiratory_rate,
           CASE WHEN v.sao2 BETWEEN 50 AND 100 THEN v.sao2 END spo2,
           CASE WHEN v.temperature BETWEEN 25 AND 45 THEN v.temperature END temperature_c,
           CASE WHEN v.systemicmean BETWEEN 20 AND 200 THEN v.systemicmean END arterial_map
    FROM cohort c
    JOIN patient_units u USING(patienthealthsystemstayid)
    JOIN vp v ON v.patientunitstayid=u.patientunitstayid
    WHERE try_cast(v.observationoffset AS BIGINT)-u.hospitaladmitoffset
          BETWEEN c.episode_end_minute-360 AND c.episode_end_minute
    """
)
con.execute(
    r"""
    CREATE TEMP TABLE aperiodic_window AS
    SELECT c.patienthealthsystemstayid, c.episode_end_minute, v.source_row_number,
           try_cast(v.observationoffset AS BIGINT)-u.hospitaladmitoffset hospital_minute,
           CASE WHEN v.noninvasivemean BETWEEN 20 AND 200 THEN v.noninvasivemean END nibp_map
    FROM cohort c
    JOIN patient_units u USING(patienthealthsystemstayid)
    JOIN va v ON v.patientunitstayid=u.patientunitstayid
    WHERE try_cast(v.observationoffset AS BIGINT)-u.hospitaladmitoffset
          BETWEEN c.episode_end_minute-360 AND c.episode_end_minute
    """
)
con.execute(
    r"""
    CREATE TEMP TABLE map_points AS
    WITH combined AS (
        SELECT patienthealthsystemstayid, episode_end_minute, hospital_minute, source_row_number,
               arterial_map map_value, 1 is_arterial
        FROM periodic_window WHERE arterial_map IS NOT NULL
        UNION ALL
        SELECT patienthealthsystemstayid, episode_end_minute, hospital_minute, source_row_number,
               nibp_map map_value, 0 is_arterial
        FROM aperiodic_window WHERE nibp_map IS NOT NULL
    ), ranked AS (
        SELECT *, row_number() OVER (
            PARTITION BY patienthealthsystemstayid,hospital_minute ORDER BY is_arterial DESC, source_row_number ASC
        ) rn FROM combined
    )
    SELECT * EXCLUDE(rn) FROM ranked WHERE rn=1
    """
)
con.execute(
    r"""
    CREATE TEMP TABLE vital_features AS
    WITH map_f AS (
        SELECT patienthealthsystemstayid,
               arg_max(map_value,hospital_minute) map_last,
               avg(map_value) FILTER (WHERE hospital_minute>episode_end_minute-60) map_mean_1h,
               min(map_value) FILTER (WHERE hospital_minute>episode_end_minute-60) map_min_1h,
               stddev_samp(map_value) FILTER (WHERE hospital_minute>episode_end_minute-60) map_sd_1h,
               avg(map_value) map_mean_6h, min(map_value) map_min_6h,
               stddev_samp(map_value) map_sd_6h,
               regr_slope(map_value,hospital_minute) map_slope_toward_stop_6h,
               100.0*avg(CASE WHEN map_value<65 THEN 1.0 ELSE 0.0 END)
                   FILTER (WHERE hospital_minute>episode_end_minute-60) map_below65_pct_1h,
               100.0*avg(CASE WHEN map_value<65 THEN 1.0 ELSE 0.0 END) map_below65_pct_6h,
               100.0*avg(is_arterial) map_arterial_pct_6h
        FROM map_points GROUP BY patienthealthsystemstayid
    ), periodic_f AS (
        SELECT patienthealthsystemstayid,
               arg_max(heart_rate,struct_pack(minute := hospital_minute, reverse_row := -source_row_number)) FILTER (WHERE heart_rate IS NOT NULL) heart_rate_last,
               avg(heart_rate) FILTER (WHERE hospital_minute>episode_end_minute-60) heart_rate_mean_1h,
               stddev_samp(heart_rate) FILTER (WHERE hospital_minute>episode_end_minute-60) heart_rate_sd_1h,
               avg(heart_rate) heart_rate_mean_6h, stddev_samp(heart_rate) heart_rate_sd_6h,
               regr_slope(heart_rate,hospital_minute) FILTER (WHERE heart_rate IS NOT NULL) heart_rate_slope_toward_stop_6h,
               arg_max(spo2,struct_pack(minute := hospital_minute, reverse_row := -source_row_number)) FILTER (WHERE spo2 IS NOT NULL) spo2_last,
               avg(spo2) spo2_mean_6h,
               arg_max(respiratory_rate,struct_pack(minute := hospital_minute, reverse_row := -source_row_number)) FILTER (WHERE respiratory_rate IS NOT NULL) respiratory_rate_last,
               avg(respiratory_rate) respiratory_rate_mean_6h,
               arg_max(temperature_c,struct_pack(minute := hospital_minute, reverse_row := -source_row_number)) FILTER (WHERE temperature_c IS NOT NULL) temperature_c_last
        FROM periodic_window GROUP BY patienthealthsystemstayid
    )
    SELECT p.*, m.* EXCLUDE(patienthealthsystemstayid)
    FROM periodic_f p FULL JOIN map_f m USING(patienthealthsystemstayid)
    """
)

con.execute(
    r"""
    CREATE TEMP TABLE admission_map AS
    SELECT patienthealthsystemstayid,
           CASE
             WHEN regexp_matches(lower(coalesce(any_value(hospitaladmitsource),'')), 'emergency')
               OR regexp_matches(lower(coalesce(any_value(unitadmitsource),'')), 'emergency') THEN 'EW EMER.'
             WHEN regexp_matches(lower(coalesce(any_value(hospitaladmitsource),'')), 'operating room|surgery')
               OR regexp_matches(lower(coalesce(any_value(unitadmitsource),'')), 'operating room|surgery') THEN 'SURGICAL SAME DAY ADMISSION'
             WHEN regexp_matches(lower(coalesce(any_value(hospitaladmitsource),'')), 'clinic|physician referral') THEN 'ELECTIVE'
             ELSE 'URGENT'
           END admission_type
    FROM patient_units GROUP BY patienthealthsystemstayid
    """
)

con.execute(
    r"""
    CREATE TEMP TABLE features AS
    SELECT c.*,
           a.admission_type,
           e.distinct_drugs,
           v.* EXCLUDE(patienthealthsystemstayid),
           d.* EXCLUDE(patienthealthsystemstayid),
           CASE WHEN v.map_last IS NOT NULL AND v.map_last>0
                THEN v.heart_rate_last/v.map_last END modified_shock_index_last,
           CASE WHEN d.nee_last_5min IS NOT NULL AND d.nee_max_6h>0
                THEN d.nee_last_5min/d.nee_max_6h END nee_last_to_max_ratio
    FROM cohort c
    LEFT JOIN admission_map a USING(patienthealthsystemstayid)
    LEFT JOIN episode_descriptors e USING(patienthealthsystemstayid)
    LEFT JOIN vital_features v USING(patienthealthsystemstayid)
    LEFT JOIN dose_features d USING(patienthealthsystemstayid)
    """
)
con.execute(
    f"COPY features TO '{p(OUTPUT)}' (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 100000)"
)

coverage = con.execute(
    """
    SELECT
        count(*) stays,
        count(map_last) map_available,
        count(heart_rate_last) hr_available,
        count(nee_last_5min) nee_last_available,
        sum(CASE WHEN dose_grid_observed>=36 THEN 1 ELSE 0 END) dose_half_grid_available,
        sum(CASE WHEN dose_grid_observed>=65 THEN 1 ELSE 0 END) dose_near_complete_grid,
        count(DISTINCT hospitalid) hospitals,
        sum(competing_outcome_code=1) restart_first,
        sum(competing_outcome_code=2) death_first
    FROM features
    """
).fetchdf()
coverage.to_csv(SUMMARY, index=False)
print("EICU STAGE 2 EXTERNAL FEATURES COMPLETED")
print(coverage.to_string(index=False))
print(f"\nFeatures: {OUTPUT}")
print(f"Coverage: {SUMMARY}")
con.close()

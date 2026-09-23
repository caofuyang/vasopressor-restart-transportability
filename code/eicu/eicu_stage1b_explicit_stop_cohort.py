from pathlib import Path
import os

import duckdb


ROOT = Path(os.environ["EICU_RAW"])
DERIVED = Path(os.environ["EICU_DERIVED"])
OUT = DERIVED / "stage1b_explicit_stop_cohort.parquet"
SUMMARY = DERIVED / "stage1b_explicit_stop_summary.csv"


def p(path: Path) -> str:
    return str(path).replace("\\", "/").replace("'", "''")


con = duckdb.connect()
con.execute("PRAGMA threads=1")
con.execute("PRAGMA memory_limit='8GB'")
con.execute("SET temp_directory='" + os.environ["REPRO_TEMP"].replace("\\", "/").replace("'", "''") + "'")
con.execute(f"CREATE VIEW patient AS SELECT * FROM read_csv_auto('{p(ROOT/'patient.csv.gz')}', header=true)")
con.execute(
    f"CREATE VIEW infusion AS SELECT * FROM read_csv_auto('{p(ROOT/'infusionDrug.csv.gz')}', "
    "header=true, all_varchar=true)"
)
con.execute(f"CREATE VIEW eol AS SELECT * FROM read_csv_auto('{p(ROOT/'carePlanEOL.csv.gz')}', header=true)")

con.execute(
    r"""
    CREATE TEMP TABLE events AS
    SELECT
        p.patienthealthsystemstayid,
        try_cast(i.infusionoffset AS BIGINT)-p.hospitaladmitoffset hospital_minute,
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
        try_cast(i.infusiondrugid AS BIGINT) infusiondrugid
    FROM infusion i JOIN patient p
      ON try_cast(i.patientunitstayid AS BIGINT)=p.patientunitstayid
    WHERE (try_cast(p.age AS INTEGER)>=18 OR p.age='> 89')
      AND p.hospitaladmitoffset IS NOT NULL
      AND try_cast(i.infusionoffset AS BIGINT) IS NOT NULL
      AND regexp_matches(lower(i.drugname),
          'norepinephrine|noradrenaline|levophed|phenylephrine|neo[- ]?synephrine|vasopressin|epinephrine|adrenaline|dopamine|angiotensin')
    """
)

# Resolve multiple entries for one drug at one minute using the last source row,
# pivot to drug-specific state updates, and carry each state forward.
con.execute(
    r"""
    CREATE TEMP TABLE timeline AS
    WITH dedup AS (
        SELECT patienthealthsystemstayid,hospital_minute,drug,
               arg_max(rate,infusiondrugid) rate
        FROM events WHERE drug IS NOT NULL AND rate IS NOT NULL AND rate>=0
        GROUP BY patienthealthsystemstayid,hospital_minute,drug
    ), pivoted AS (
        SELECT patienthealthsystemstayid,hospital_minute,
          max(rate) FILTER(WHERE drug='norepinephrine') norepinephrine,
          max(rate) FILTER(WHERE drug='phenylephrine') phenylephrine,
          max(rate) FILTER(WHERE drug='vasopressin') vasopressin,
          max(rate) FILTER(WHERE drug='epinephrine') epinephrine,
          max(rate) FILTER(WHERE drug='dopamine') dopamine,
          max(rate) FILTER(WHERE drug='angiotensin_ii') angiotensin_ii
        FROM dedup GROUP BY patienthealthsystemstayid,hospital_minute
    ), filled AS (
        SELECT patienthealthsystemstayid,hospital_minute,
          last_value(norepinephrine IGNORE NULLS) OVER w norepinephrine,
          last_value(phenylephrine IGNORE NULLS) OVER w phenylephrine,
          last_value(vasopressin IGNORE NULLS) OVER w vasopressin,
          last_value(epinephrine IGNORE NULLS) OVER w epinephrine,
          last_value(dopamine IGNORE NULLS) OVER w dopamine,
          last_value(angiotensin_ii IGNORE NULLS) OVER w angiotensin_ii
        FROM pivoted
        WINDOW w AS (PARTITION BY patienthealthsystemstayid ORDER BY hospital_minute
                     ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW)
    ), active AS (
        SELECT *,
          (coalesce(norepinephrine>0,false)::INTEGER+
           coalesce(phenylephrine>0,false)::INTEGER+
           coalesce(vasopressin>0,false)::INTEGER+
           coalesce(epinephrine>0,false)::INTEGER+
           coalesce(dopamine>0,false)::INTEGER+
           coalesce(angiotensin_ii>0,false)::INTEGER) active_agents
        FROM filled
    )
    SELECT *, lag(active_agents,1,0) OVER(
        PARTITION BY patienthealthsystemstayid ORDER BY hospital_minute
    ) previous_active_agents
    FROM active
    """
)

con.execute(
    r"""
    CREATE TEMP TABLE transitions AS
    SELECT *,
      CASE WHEN previous_active_agents=0 AND active_agents>0 THEN 1 ELSE 0 END is_start,
      CASE WHEN previous_active_agents>0 AND active_agents=0 THEN 1 ELSE 0 END is_stop,
      sum(CASE WHEN previous_active_agents=0 AND active_agents>0 THEN 1 ELSE 0 END) OVER(
        PARTITION BY patienthealthsystemstayid ORDER BY hospital_minute
        ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
      ) episode_number
    FROM timeline
    """
)
con.execute(
    r"""
    CREATE TEMP TABLE episodes AS
    SELECT patienthealthsystemstayid,episode_number,
      min(hospital_minute) FILTER(WHERE is_start=1) episode_start_minute,
      min(hospital_minute) FILTER(WHERE is_stop=1) episode_end_minute,
      arg_max(
        CASE
          WHEN norepinephrine>0 THEN 'norepinephrine'
          WHEN phenylephrine>0 THEN 'phenylephrine'
          WHEN vasopressin>0 THEN 'vasopressin'
          WHEN epinephrine>0 THEN 'epinephrine'
          WHEN dopamine>0 THEN 'dopamine'
          WHEN angiotensin_ii>0 THEN 'angiotensin_ii'
        END,
        hospital_minute
      ) FILTER(WHERE active_agents>0) terminal_drug,
      max(previous_active_agents) FILTER(WHERE is_stop=1) active_agents_before_stop
    FROM transitions
    WHERE episode_number>0
    GROUP BY patienthealthsystemstayid,episode_number
    HAVING episode_start_minute IS NOT NULL AND episode_end_minute IS NOT NULL
    """
)

con.execute(
    r"""
    CREATE TEMP TABLE patient_one AS
    SELECT patienthealthsystemstayid, any_value(uniquepid) uniquepid,
      any_value(gender) gender,
      CASE WHEN max(age='> 89') THEN 90 ELSE max(try_cast(age AS INTEGER)) END age,
      any_value(hospitalid) hospitalid,
      any_value(hospitaldischargestatus) hospitaldischargestatus,
      max(hospitaldischargeoffset-hospitaladmitoffset) hospital_discharge_minute,
      any_value(unitadmitsource) unitadmitsource,
      any_value(hospitaladmitsource) hospitaladmitsource
    FROM patient GROUP BY patienthealthsystemstayid
    """
)

con.execute(
    r"""
    CREATE TEMP TABLE candidates AS
    WITH sequenced AS (
      SELECT *, lead(episode_start_minute) OVER(
        PARTITION BY patienthealthsystemstayid ORDER BY episode_number
      ) next_episode_start_minute
      FROM episodes
    ), eligible AS (
      SELECT s.*,p.* EXCLUDE(patienthealthsystemstayid),
        (episode_end_minute-episode_start_minute)/60.0 duration_hours,
        CASE WHEN next_episode_start_minute>episode_end_minute
                  AND next_episode_start_minute<=episode_end_minute+1440 THEN 1 ELSE 0 END restart_24h,
        CASE WHEN lower(coalesce(hospitaldischargestatus,''))='expired'
                  AND hospital_discharge_minute>episode_end_minute
                  AND hospital_discharge_minute<=episode_end_minute+1440 THEN 1 ELSE 0 END death_24h,
        row_number() OVER(PARTITION BY s.patienthealthsystemstayid ORDER BY episode_end_minute) candidate_number
      FROM sequenced s JOIN patient_one p USING(patienthealthsystemstayid)
      WHERE episode_end_minute-episode_start_minute>=360
    )
    SELECT * FROM eligible WHERE candidate_number=1
    """
)

con.execute(
    r"""
    CREATE TEMP TABLE eol_flags AS
    SELECT c.patienthealthsystemstayid,
      max(CASE WHEN e.cpleoldiscussionoffset-p.hospitaladmitoffset<=c.episode_end_minute
               THEN 1 ELSE 0 END) any_eol_discussion_before_cessation
    FROM candidates c JOIN patient p USING(patienthealthsystemstayid)
    JOIN eol e USING(patientunitstayid)
    GROUP BY c.patienthealthsystemstayid
    """
)

con.execute(
    r"""
    CREATE TEMP TABLE final AS
    SELECT c.*,coalesce(e.any_eol_discussion_before_cessation,0) any_eol_discussion_before_cessation,
      CASE WHEN restart_24h=1 AND death_24h=1 AND next_episode_start_minute<=hospital_discharge_minute THEN 1
           WHEN restart_24h=1 THEN 1 WHEN death_24h=1 THEN 2 ELSE 0 END competing_outcome_code,
      CASE WHEN restart_24h=1 AND death_24h=1 AND next_episode_start_minute<=hospital_discharge_minute THEN 'restart_first'
           WHEN restart_24h=1 THEN 'restart_first' WHEN death_24h=1 THEN 'death_first'
           ELSE 'durable_liberation_alive' END competing_outcome
    FROM candidates c LEFT JOIN eol_flags e USING(patienthealthsystemstayid)
    """
)

con.execute(f"COPY final TO '{p(OUT)}' (FORMAT PARQUET,COMPRESSION ZSTD,ROW_GROUP_SIZE 100000)")
summary=con.execute("""
SELECT competing_outcome,count(*) stays,count(DISTINCT uniquepid) patients,
 count(DISTINCT hospitalid) hospitals,round(100.0*count(*)/sum(count(*)) OVER(),2) outcome_pct,
 round(median(duration_hours),2) median_duration_hours
FROM final GROUP BY competing_outcome ORDER BY min(competing_outcome_code)
""").fetchdf()
summary.to_csv(SUMMARY,index=False)
print('EICU STAGE 1B EXPLICIT-STOP COHORT COMPLETED')
print(summary.to_string(index=False))
print('\nQA')
print(con.execute("""SELECT count(*) stays,sum(restart_24h) restarts,sum(death_24h) deaths,
 sum(restart_24h=1 AND death_24h=1) both_events,
 sum(any_eol_discussion_before_cessation) prior_eol_discussion
 FROM final""").fetchdf().to_string(index=False))
print(f'\nCohort: {OUT}')
con.close()

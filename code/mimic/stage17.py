import os
from pathlib import Path
import duckdb
derived = Path(os.environ['MIMIC_DERIVED'])
raw = Path(os.environ['MIMIC_RAW'])
cohort = derived / 'vasopressor_liberation_competing_outcomes_stage13.parquet'
vasopressors = derived / 'vasopressor_inputevents_nee_clean_stage6.parquet'
patients = raw / 'hosp' / 'patients.csv.gz'
output = derived / 'stage17_cross_icu_restart_audit.parquet'

def sql_path(path):
    return str(path).replace('\\', '/').replace("'", "''")
con = duckdb.connect()
con.execute('PRAGMA threads=1')
con.execute("PRAGMA memory_limit='8GB'")
con.execute("SET temp_directory = ?", [os.environ["REPRO_TEMP"]])
con.execute(f"\nCREATE OR REPLACE TEMP TABLE stage17 AS\nWITH index_cohort AS (\n    SELECT\n        c.*,\n        p.anchor_year_group\n    FROM read_parquet('{sql_path(cohort)}') c\n    INNER JOIN read_csv_auto(\n        '{sql_path(patients)}',\n        header=true\n    ) p\n        ON c.subject_id = p.subject_id\n),\ncross_icu AS (\n    SELECT\n        c.stay_id AS index_stay_id,\n        min(v.starttime) AS cross_icu_restart_time,\n        arg_min(v.stay_id, v.starttime) AS subsequent_stay_id,\n        arg_min(v.hadm_id, v.starttime) AS subsequent_hadm_id\n    FROM index_cohort c\n    INNER JOIN read_parquet('{sql_path(vasopressors)}') v\n        ON c.subject_id = v.subject_id\n       AND c.stay_id <> v.stay_id\n       AND v.starttime >= c.episode_end\n       AND v.starttime <= c.episode_end + INTERVAL 24 HOUR\n    GROUP BY c.stay_id\n),\ncombined AS (\n    SELECT\n        c.*,\n        x.cross_icu_restart_time,\n        x.subsequent_stay_id,\n        x.subsequent_hadm_id,\n\n        CASE\n            WHEN x.cross_icu_restart_time IS NOT NULL\n             AND c.restart_24h = 0\n            THEN 1 ELSE 0\n        END AS newly_detected_cross_icu_restart,\n\n        CASE\n            WHEN x.cross_icu_restart_time IS NOT NULL\n             AND x.subsequent_hadm_id = c.hadm_id\n            THEN 1 ELSE 0\n        END AS same_hospitalization_cross_icu_restart,\n\n        CASE\n            WHEN c.restart_24h = 1\n             AND x.cross_icu_restart_time IS NOT NULL\n            THEN least(\n                c.next_episode_start,\n                x.cross_icu_restart_time\n            )\n            WHEN c.restart_24h = 1\n            THEN c.next_episode_start\n            ELSE x.cross_icu_restart_time\n        END AS revised_restart_time\n\n    FROM index_cohort c\n    LEFT JOIN cross_icu x\n        ON c.stay_id = x.index_stay_id\n)\nSELECT\n    *,\n\n    CASE\n        WHEN revised_restart_time IS NOT NULL\n        THEN 1 ELSE 0\n    END AS revised_restart_24h,\n\n    CASE\n        WHEN revised_restart_time IS NULL\n         AND death_24h = 0\n        THEN 'durable_liberation_alive'\n\n        WHEN revised_restart_time IS NOT NULL\n         AND death_24h = 0\n        THEN 'restart_first'\n\n        WHEN revised_restart_time IS NULL\n         AND death_24h = 1\n        THEN 'death_first'\n\n        WHEN revised_restart_time IS NOT NULL\n         AND death_24h = 1\n         AND revised_restart_time <= deathtime\n        THEN 'restart_first'\n\n        WHEN revised_restart_time IS NOT NULL\n         AND death_24h = 1\n         AND deathtime < revised_restart_time\n        THEN 'death_first'\n\n        ELSE 'unclassifiable'\n    END AS revised_competing_outcome\n\nFROM combined\n")
con.execute(f"\nCOPY stage17 TO '{sql_path(output)}'\n(FORMAT PARQUET, COMPRESSION ZSTD)\n")
qa = con.execute("\nSELECT\n    count(*) AS development_validation_stays,\n    sum(early_alive_icu_exit) AS early_alive_icu_exits,\n    sum(restart_24h) AS original_restarts,\n    sum(newly_detected_cross_icu_restart)\n        AS newly_detected_cross_icu_restarts,\n    sum(\n        newly_detected_cross_icu_restart *\n        early_alive_icu_exit\n    ) AS new_restarts_among_early_exits,\n    sum(same_hospitalization_cross_icu_restart)\n        AS same_hospitalization_cross_icu_restarts\nFROM stage17\nWHERE anchor_year_group <> '2020 - 2022'\n").fetchdf()
comparison = con.execute("\nWITH old_outcomes AS (\n    SELECT\n        'Original outcome' AS definition,\n        competing_outcome AS outcome,\n        count(*) AS stays\n    FROM stage17\n    WHERE anchor_year_group <> '2020 - 2022'\n    GROUP BY competing_outcome\n),\nnew_outcomes AS (\n    SELECT\n        'Cross-ICU corrected' AS definition,\n        revised_competing_outcome AS outcome,\n        count(*) AS stays\n    FROM stage17\n    WHERE anchor_year_group <> '2020 - 2022'\n    GROUP BY revised_competing_outcome\n)\nSELECT * FROM old_outcomes\nUNION ALL\nSELECT * FROM new_outcomes\nORDER BY definition, outcome\n").fetchdf()
new_restart_details = con.execute("\nSELECT\n    early_alive_icu_exit,\n    count(*) AS newly_detected_restarts,\n    sum(\n        CASE\n            WHEN subsequent_hadm_id = hadm_id\n            THEN 1 ELSE 0\n        END\n    ) AS same_hospitalization,\n    round(\n        median(\n            date_diff(\n                'second',\n                episode_end,\n                cross_icu_restart_time\n            ) / 3600.0\n        ),\n        2\n    ) AS median_hours_after_cessation\nFROM stage17\nWHERE anchor_year_group <> '2020 - 2022'\n  AND newly_detected_cross_icu_restart = 1\nGROUP BY early_alive_icu_exit\nORDER BY early_alive_icu_exit\n").fetchdf()
print('\nSTAGE 17 CROSS-ICU RESTART AUDIT COMPLETED')
print('\nQA - DEVELOPMENT AND TEMPORAL VALIDATION ONLY')
print(qa.to_string(index=False))
print('\nORIGINAL VS CORRECTED OUTCOMES')
print(comparison.to_string(index=False))
print('\nNEW CROSS-ICU RESTART DETAILS')
if len(new_restart_details) == 0:
    print('No newly detected cross-ICU restarts.')
else:
    print(new_restart_details.to_string(index=False))
print(f'\nOutput: {output}')
print('No locked-period model predictions were evaluated.')
con.close()

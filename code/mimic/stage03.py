import os
from pathlib import Path
import duckdb
root = Path(os.environ['MIMIC_RAW'])
stage2 = Path(os.environ['MIMIC_DERIVED']) / 'vasopressor_liberation_attempts_stage2.parquet'
output = Path(os.environ['MIMIC_DERIVED']) / 'vasopressor_liberation_primary_cohort_stage3.parquet'
temp = Path(os.environ['REPRO_TEMP'])
if output.exists():
    raise SystemExit(f'输出已存在，未覆盖：{output}')

def sql_path(p):
    return str(p).replace('\\', '/').replace("'", "''")
con = duckdb.connect()
con.execute('SET threads = 1')
con.execute("SET memory_limit = '8GB'")
con.execute(f"SET temp_directory = '{sql_path(temp)}'")
con.execute(f"\nCREATE TEMP TABLE primary_cohort AS\nWITH enriched AS (\n    SELECT\n        c.*,\n        p.gender,\n        p.anchor_age\n          + year(a.admittime)\n          - p.anchor_year AS age,\n        a.race,\n        a.admission_type,\n        a.insurance,\n        a.marital_status\n    FROM read_parquet('{sql_path(stage2)}') AS c\n    INNER JOIN read_csv_auto(\n        '{sql_path(root / 'hosp' / 'patients.csv.gz')}',\n        header=true, sample_size=1000000\n    ) AS p\n        ON c.subject_id = p.subject_id\n    INNER JOIN read_csv_auto(\n        '{sql_path(root / 'hosp' / 'admissions.csv.gz')}',\n        header=true, sample_size=1000000\n    ) AS a\n        ON c.subject_id = a.subject_id\n       AND c.hadm_id = a.hadm_id\n    WHERE c.terminal_status = 'Stopped'\n),\nadult_stopped AS (\n    SELECT *,\n        row_number() OVER (\n            PARTITION BY stay_id\n            ORDER BY episode_end\n        ) AS stopped_attempt_number\n    FROM enriched\n    WHERE age >= 18\n)\nSELECT *\nFROM adult_stopped\nWHERE stopped_attempt_number = 1\n")
con.execute(f"\nCOPY primary_cohort\nTO '{sql_path(output)}'\n(FORMAT PARQUET, COMPRESSION ZSTD)\n")
overall = con.execute("\nSELECT\n    count(*) AS icu_stays,\n    count(DISTINCT subject_id) AS patients,\n    round(median(age), 1) AS median_age,\n    round(100.0 * avg(CASE WHEN gender = 'F' THEN 1 ELSE 0 END), 2)\n        AS female_pct,\n    sum(restart_24h) AS restart_24h,\n    sum(death_24h) AS death_24h,\n    sum(failure_24h) AS composite_failure_24h,\n    round(100.0 * avg(failure_24h), 2) AS failure_pct,\n    sum(strict_24h_observable) AS strict_observable,\n    sum(early_alive_icu_exit) AS early_alive_icu_exit\nFROM primary_cohort\n").fetchdf()
by_drug = con.execute('\nSELECT\n    terminal_drug,\n    count(*) AS attempts,\n    sum(failure_24h) AS failures,\n    round(100.0 * avg(failure_24h), 2) AS failure_pct\nFROM primary_cohort\nGROUP BY terminal_drug\nORDER BY attempts DESC\n').fetchdf()
by_admission = con.execute('\nSELECT\n    admission_type,\n    count(*) AS attempts,\n    sum(failure_24h) AS failures,\n    round(100.0 * avg(failure_24h), 2) AS failure_pct\nFROM primary_cohort\nGROUP BY admission_type\nORDER BY attempts DESC\n').fetchdf()
print('STAGE 3 PRIMARY COHORT COMPLETED')
print('\nOVERALL')
print(overall.to_string(index=False))
print('\nBY TERMINAL DRUG')
print(by_drug.to_string(index=False))
print('\nBY ADMISSION TYPE')
print(by_admission.to_string(index=False))
print(f'\nOutput: {output}')
con.close()

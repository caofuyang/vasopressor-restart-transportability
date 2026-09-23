import os
from pathlib import Path
import duckdb
d = Path(os.environ['MIMIC_DERIVED'])
temp = Path(os.environ['REPRO_TEMP'])
cohort = d / 'vasopressor_liberation_primary_cohort_stage3.parquet'
vitals = d / 'preliberation_vitals_6h_stage4.parquet'
labs = d / 'preliberation_labs_6h_stage4.parquet'
vitals_clean = d / 'preliberation_vitals_6h_clean_stage5.parquet'
labs_clean = d / 'preliberation_labs_6h_clean_stage5.parquet'
for path in (vitals_clean, labs_clean):
    if path.exists():
        raise SystemExit(f'输出已存在，未覆盖：{path}')

def sql_path(p):
    return str(p).replace('\\', '/').replace("'", "''")
con = duckdb.connect()
con.execute('SET threads = 1')
con.execute("SET memory_limit = '8GB'")
con.execute(f"SET temp_directory = '{sql_path(temp)}'")
con.execute(f"\nCOPY (\n    SELECT *\n    FROM read_parquet('{sql_path(vitals)}')\n    WHERE\n        (variable = 'heart_rate' AND valuenum BETWEEN 20 AND 250)\n     OR (variable = 'respiratory_rate' AND valuenum BETWEEN 2 AND 80)\n     OR (variable = 'spo2' AND valuenum BETWEEN 50 AND 100)\n     OR (variable = 'temperature_c' AND valuenum BETWEEN 25 AND 45)\n     OR (variable IN ('map_arterial', 'map_noninvasive')\n         AND valuenum BETWEEN 20 AND 200)\n)\nTO '{sql_path(vitals_clean)}'\n(FORMAT PARQUET, COMPRESSION ZSTD)\n")
con.execute(f"\nCOPY (\n    SELECT *\n    FROM read_parquet('{sql_path(labs)}')\n    WHERE\n        (variable = 'ph' AND valuenum BETWEEN 6.5 AND 8.0)\n     OR (variable = 'base_excess' AND valuenum BETWEEN -40 AND 40)\n     OR (variable = 'bicarbonate' AND valuenum BETWEEN 5 AND 60)\n     OR (variable = 'creatinine' AND valuenum BETWEEN 0.1 AND 20)\n     OR (variable = 'hemoglobin' AND valuenum BETWEEN 3 AND 25)\n     OR (variable = 'platelets' AND valuenum BETWEEN 5 AND 2000)\n     OR (variable = 'wbc' AND valuenum BETWEEN 0.1 AND 200)\n     OR (variable = 'lactate' AND valuenum BETWEEN 0.2 AND 30)\n     OR (variable = 'bilirubin_total' AND valuenum BETWEEN 0.1 AND 80)\n     OR (variable = 'albumin' AND valuenum BETWEEN 0.5 AND 6)\n)\nTO '{sql_path(labs_clean)}'\n(FORMAT PARQUET, COMPRESSION ZSTD)\n")
n = con.execute(f"SELECT count(*) FROM read_parquet('{sql_path(cohort)}')").fetchone()[0]
vital_rows = con.execute(f"\nSELECT\n    variable,\n    count(*) AS clean_records,\n    count(DISTINCT stay_id) AS covered_stays,\n    round(100.0 * count(DISTINCT stay_id) / {n}, 2) AS coverage_pct\nFROM read_parquet('{sql_path(vitals_clean)}')\nGROUP BY variable\nORDER BY coverage_pct DESC\n").fetchall()
lab_rows = con.execute(f"\nSELECT\n    variable,\n    count(*) AS clean_records,\n    count(DISTINCT stay_id) AS covered_stays,\n    round(100.0 * count(DISTINCT stay_id) / {n}, 2) AS coverage_pct\nFROM read_parquet('{sql_path(labs_clean)}')\nGROUP BY variable\nORDER BY coverage_pct DESC\n").fetchall()
core_rows = []
for minutes in (60, 180, 360):
    row = con.execute(f"\n    WITH flags AS (\n        SELECT\n            c.stay_id,\n            max(CASE\n                WHEN v.variable = 'heart_rate'\n                 AND v.minutes_before_stop BETWEEN 0 AND {minutes}\n                THEN 1 ELSE 0 END\n            ) AS has_hr,\n            max(CASE\n                WHEN v.variable IN ('map_arterial', 'map_noninvasive')\n                 AND v.minutes_before_stop BETWEEN 0 AND {minutes}\n                THEN 1 ELSE 0 END\n            ) AS has_map\n        FROM read_parquet('{sql_path(cohort)}') AS c\n        LEFT JOIN read_parquet('{sql_path(vitals_clean)}') AS v\n            ON c.stay_id = v.stay_id\n        GROUP BY c.stay_id\n    )\n    SELECT\n        {minutes} AS window_minutes,\n        count(*) FILTER (WHERE has_hr = 1 AND has_map = 1)\n            AS stays_with_hr_and_map,\n        round(\n            100.0 * count(*) FILTER (WHERE has_hr = 1 AND has_map = 1)\n            / count(*), 2\n        ) AS coverage_pct\n    FROM flags\n    ").fetchone()
    core_rows.append(row)
print('STAGE 5 COMPLETED')
print('VITAL CLEAN:', '; '.join((f'{r[0]}={r[3]}% ({r[2]}/{n})' for r in vital_rows)))
print('LAB CLEAN:', '; '.join((f'{r[0]}={r[3]}% ({r[2]}/{n})' for r in lab_rows)))
print('HR+MAP COVERAGE:', '; '.join((f'{r[0]}min={r[2]}% ({r[1]}/{n})' for r in core_rows)))
print(f'Vitals clean: {vitals_clean}')
print(f'Labs clean: {labs_clean}')
con.close()

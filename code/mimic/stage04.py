import os
from pathlib import Path
import duckdb
from stream_raw import vitals as scan_vitals, labs as scan_labs
root = Path(os.environ['MIMIC_RAW'])
derived = Path(os.environ['MIMIC_DERIVED'])
temp = Path(os.environ['REPRO_TEMP'])
cohort = derived / 'vasopressor_liberation_primary_cohort_stage3.parquet'
vitals_out = derived / 'preliberation_vitals_6h_stage4.parquet'
labs_out = derived / 'preliberation_labs_6h_stage4.parquet'
for path in (vitals_out, labs_out):
    if path.exists():
        raise SystemExit(f'输出已存在，未覆盖：{path}')

def sql_path(p):
    return str(p).replace('\\', '/').replace("'", "''")
raw_vitals = temp / 'stage04_raw_vitals.parquet'
raw_labs = temp / 'stage04_raw_labs.parquet'
scan_vitals(root / 'icu/chartevents.csv.gz', raw_vitals, cohort)
scan_labs(root / 'hosp/labevents.csv.gz', raw_labs, cohort)
con = duckdb.connect()
con.execute('SET threads = 1')
con.execute("SET memory_limit = '8GB'")
con.execute(f"SET temp_directory = '{sql_path(temp)}'")
con.execute('SET preserve_insertion_order = false')
con.execute(f"\nCREATE TEMP TABLE lab_items AS\nSELECT *\nFROM (\n    SELECT\n        itemid,\n        label,\n        CASE\n            WHEN lower(label) = 'lactate' THEN 'lactate'\n            WHEN lower(label) = 'creatinine' THEN 'creatinine'\n            WHEN lower(label) = 'bicarbonate' THEN 'bicarbonate'\n            WHEN lower(label) = 'base excess' THEN 'base_excess'\n            WHEN lower(label) = 'ph' THEN 'ph'\n            WHEN lower(label) = 'hemoglobin' THEN 'hemoglobin'\n            WHEN lower(label) = 'platelet count' THEN 'platelets'\n            WHEN lower(label) IN ('white blood cells', 'wbc count')\n                THEN 'wbc'\n            WHEN lower(label) = 'albumin' THEN 'albumin'\n            WHEN lower(label) IN ('bilirubin, total', 'bilirubin total')\n                THEN 'bilirubin_total'\n        END AS variable\n    FROM read_csv_auto(\n        '{sql_path(root / 'hosp' / 'd_labitems.csv.gz')}',\n        header=true\n    )\n    WHERE lower(coalesce(fluid, '')) = 'blood'\n)\nWHERE variable IS NOT NULL\n")
print('开始扫描 chartevents，可能需要较长时间……')
con.execute(f"\nCOPY (\n    SELECT\n        c.subject_id,\n        c.hadm_id,\n        c.stay_id,\n        c.episode_end,\n        v.source_row_number,\n        v.charttime,\n        v.itemid,\n        CASE\n            WHEN v.itemid = 220052 THEN 'map_arterial'\n            WHEN v.itemid = 220181 THEN 'map_noninvasive'\n            WHEN v.itemid = 220045 THEN 'heart_rate'\n            WHEN v.itemid = 220277 THEN 'spo2'\n            WHEN v.itemid IN (220210, 224690) THEN 'respiratory_rate'\n            WHEN v.itemid IN (223761, 223762) THEN 'temperature_c'\n        END AS variable,\n        CASE\n            WHEN v.itemid = 223761\n                THEN (v.valuenum - 32.0) * 5.0 / 9.0\n            ELSE v.valuenum\n        END AS valuenum,\n        CASE\n            WHEN v.itemid IN (223761, 223762) THEN 'degC'\n            ELSE v.valueuom\n        END AS valueuom,\n        date_diff('minute', v.charttime, c.episode_end)\n            AS minutes_before_stop\n    FROM read_parquet('{sql_path(raw_vitals)}') AS v\n    INNER JOIN read_parquet('{sql_path(cohort)}') AS c\n        ON v.stay_id = c.stay_id\n       AND v.charttime > c.episode_end - INTERVAL '6 hours'\n       AND v.charttime <= c.episode_end\n    WHERE v.itemid IN\n        (220052,220181,220045,220277,220210,224690,223761,223762)\n      AND v.valuenum IS NOT NULL\n    ORDER BY v.source_row_number, c.stay_id\n)\nTO '{sql_path(vitals_out)}'\n(FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 250000)\n")
print('生命体征提取完成，开始扫描 labevents……')
con.execute(f"\nCOPY (\n    SELECT\n        c.subject_id,\n        c.hadm_id,\n        c.stay_id,\n        c.episode_end,\n        l.source_row_number,\n        l.charttime,\n        l.itemid,\n        d.variable,\n        l.valuenum,\n        l.valueuom,\n        date_diff('minute', l.charttime, c.episode_end)\n            AS minutes_before_stop\n    FROM read_parquet('{sql_path(raw_labs)}') AS l\n    INNER JOIN read_parquet('{sql_path(cohort)}') AS c\n        ON l.hadm_id = c.hadm_id\n       AND l.charttime > c.episode_end - INTERVAL '6 hours'\n       AND l.charttime <= c.episode_end\n    INNER JOIN lab_items AS d\n        ON l.itemid = d.itemid\n    WHERE l.valuenum IS NOT NULL\n)\nTO '{sql_path(labs_out)}'\n(FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 250000)\n")
denominator = con.execute(f"SELECT count(*) FROM read_parquet('{sql_path(cohort)}')").fetchone()[0]
vital_rows = con.execute(f"\nSELECT\n    variable,\n    count(*) AS records,\n    count(DISTINCT stay_id) AS stays,\n    round(100.0 * count(DISTINCT stay_id) / {denominator}, 2)\n        AS coverage_pct,\n    min(valuenum) AS raw_min,\n    max(valuenum) AS raw_max\nFROM read_parquet('{sql_path(vitals_out)}')\nGROUP BY variable\nORDER BY coverage_pct DESC\n").fetchall()
lab_rows = con.execute(f"\nSELECT\n    variable,\n    count(*) AS records,\n    count(DISTINCT stay_id) AS stays,\n    round(100.0 * count(DISTINCT stay_id) / {denominator}, 2)\n        AS coverage_pct,\n    min(valuenum) AS raw_min,\n    max(valuenum) AS raw_max\nFROM read_parquet('{sql_path(labs_out)}')\nGROUP BY variable\nORDER BY coverage_pct DESC\n").fetchall()
print('STAGE 4 COMPLETED')
print('VITAL:', '; '.join((f'{r[0]}={r[2]}% ({r[1]} records; raw {r[3]}-{r[4]})' for r in vital_rows)))
print('LAB:', '; '.join((f'{r[0]}={r[2]}% ({r[1]} records; raw {r[3]}-{r[4]})' for r in lab_rows)))
print(f'Vitals: {vitals_out}')
print(f'Labs: {labs_out}')
con.close()

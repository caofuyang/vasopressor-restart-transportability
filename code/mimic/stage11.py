import os
from pathlib import Path
import duckdb
from stream_raw import vitals as scan_vitals, labs as scan_labs
root = Path(os.environ['MIMIC_RAW'])
d = Path(os.environ['MIMIC_DERIVED'])
temp = Path(os.environ['REPRO_TEMP'])
cohort = d / 'vasopressor_liberation_primary_cohort_stage3.parquet'
output = d / 'preliberation_code_status_stage11.parquet'
if output.exists():
    raise SystemExit(f'输出已存在，未覆盖：{output}')

def sql_path(p):
    return str(p).replace('\\', '/').replace("'", "''")
raw_codes = temp / 'stage11_raw_codes.parquet'
scan_vitals(root / 'icu/chartevents.csv.gz', raw_codes, cohort, code_status=True)
con = duckdb.connect()
con.execute('SET threads = 1')
con.execute("SET memory_limit = '8GB'")
con.execute(f"SET temp_directory = '{sql_path(temp)}'")
con.execute('SET preserve_insertion_order = false')
print('开始扫描 Code Status；可能需要较长时间……')
con.execute(f"\nCOPY (\n    SELECT\n        c.subject_id,\n        c.hadm_id,\n        c.stay_id,\n        c.intime,\n        c.episode_end,\n        c.death_24h,\n        ce.source_row_number,\n        ce.charttime,\n        ce.itemid,\n        ce.value,\n        date_diff('minute', ce.charttime, c.episode_end)\n            AS minutes_before_stop\n    FROM read_parquet('{sql_path(raw_codes)}') AS ce\n    INNER JOIN read_parquet('{sql_path(cohort)}') AS c\n        ON ce.stay_id = c.stay_id\n       AND ce.charttime >= c.intime\n       AND ce.charttime <= c.episode_end + INTERVAL '1 hour'\n    WHERE ce.itemid IN (223758, 229784, 228687)\n      AND ce.value IS NOT NULL\n    ORDER BY ce.source_row_number, c.stay_id\n)\nTO '{sql_path(output)}'\n(FORMAT PARQUET, COMPRESSION ZSTD)\n")
distribution = con.execute(f"\nSELECT\n    itemid,\n    value,\n    count(*) AS records,\n    count(DISTINCT stay_id) AS stays,\n    count(DISTINCT stay_id) FILTER (\n        WHERE minutes_before_stop >= 0\n    ) AS stays_documented_before_stop,\n    count(DISTINCT stay_id) FILTER (\n        WHERE death_24h = 1\n    ) AS death_24h_stays\nFROM read_parquet('{sql_path(output)}')\nGROUP BY itemid, value\nORDER BY stays DESC, itemid, value\n").fetchall()
coverage = con.execute(f"\nSELECT\n    count(*) AS records,\n    count(DISTINCT stay_id) AS covered_stays\nFROM read_parquet('{sql_path(output)}')\n").fetchone()
print('STAGE 11 CODE STATUS EXTRACTION COMPLETED')
print(f'Coverage: {coverage[1]}/12016 stays; {coverage[0]} records')
print('VALUE DISTRIBUTION')
for r in distribution:
    print(f'itemid={r[0]} | value={r[1]} | records={r[2]} | stays={r[3]} | before_stop={r[4]} | death24={r[5]}')
print(f'Output: {output}')
con.close()

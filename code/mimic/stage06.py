import os
from pathlib import Path
import duckdb
d = Path(os.environ['MIMIC_DERIVED'])
temp = Path(os.environ['REPRO_TEMP'])
source = d / 'vasopressor_inputevents_stage1.parquet'
output = d / 'vasopressor_inputevents_nee_clean_stage6.parquet'
if output.exists():
    raise SystemExit(f'输出已存在，未覆盖：{output}')

def sql_path(p):
    return str(p).replace('\\', '/').replace("'", "''")
con = duckdb.connect()
con.execute('SET threads = 1')
con.execute("SET memory_limit = '8GB'")
con.execute(f"SET temp_directory = '{sql_path(temp)}'")
con.execute(f"\nCOPY (\n    WITH converted AS (\n        SELECT *,\n            CASE\n                WHEN drug = 'norepinephrine'\n                 AND rateuom = 'mcg/kg/min'\n                 AND rate BETWEEN 0.0001 AND 10\n                    THEN rate\n\n                WHEN drug = 'norepinephrine'\n                 AND rateuom = 'mg/kg/min'\n                 AND rate * 1000 BETWEEN 0.0001 AND 10\n                    THEN rate * 1000\n\n                WHEN drug = 'epinephrine'\n                 AND rateuom = 'mcg/kg/min'\n                 AND rate BETWEEN 0.0001 AND 10\n                    THEN rate\n\n                WHEN drug = 'dopamine'\n                 AND rateuom = 'mcg/kg/min'\n                 AND rate BETWEEN 0.01 AND 100\n                    THEN rate / 100\n\n                WHEN drug = 'phenylephrine'\n                 AND rateuom = 'mcg/kg/min'\n                 AND rate BETWEEN 0.001 AND 20\n                    THEN rate / 10\n\n                WHEN drug = 'phenylephrine'\n                 AND rateuom = 'mcg/min'\n                 AND patientweight BETWEEN 20 AND 300\n                 AND rate / patientweight BETWEEN 0.001 AND 20\n                    THEN (rate / patientweight) / 10\n\n                WHEN drug = 'vasopressin'\n                 AND rateuom = 'units/hour'\n                 AND rate BETWEEN 0.001 AND 12\n                    THEN (rate / 60) * 2.5\n\n                WHEN drug = 'vasopressin'\n                 AND rateuom = 'units/min'\n                 AND rate BETWEEN 0.0001 AND 0.2\n                    THEN rate * 2.5\n\n                WHEN drug = 'angiotensin_ii'\n                 AND rateuom = 'ng/kg/min'\n                 AND rate BETWEEN 0.1 AND 200\n                    THEN (rate / 1000) * 10\n\n                WHEN drug = 'angiotensin_ii'\n                 AND rateuom = 'mcg/kg/min'\n                 AND rate BETWEEN 0.0001 AND 0.2\n                    THEN rate * 10\n            END AS norepinephrine_equivalent\n        FROM read_parquet('{sql_path(source)}')\n        WHERE itemid <> 229617\n          AND rate IS NOT NULL\n          AND rate > 0\n          AND endtime > starttime\n          AND statusdescription IN\n              ('ChangeDose/Rate', 'FinishedRunning', 'Stopped', 'Paused')\n    )\n    SELECT *\n    FROM converted\n    WHERE norepinephrine_equivalent IS NOT NULL\n      AND norepinephrine_equivalent > 0\n      AND norepinephrine_equivalent <= 10\n)\nTO '{sql_path(output)}'\n(FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 250000)\n")
summary = con.execute(f"\nWITH raw AS (\n    SELECT\n        drug,\n        count(*) AS raw_records\n    FROM read_parquet('{sql_path(source)}')\n    WHERE itemid <> 229617\n      AND rate IS NOT NULL\n      AND rate > 0\n      AND endtime > starttime\n      AND statusdescription IN\n          ('ChangeDose/Rate', 'FinishedRunning', 'Stopped', 'Paused')\n    GROUP BY drug\n),\nclean AS (\n    SELECT\n        drug,\n        count(*) AS clean_records,\n        quantile_cont(norepinephrine_equivalent, 0.01) AS nee_p01,\n        median(norepinephrine_equivalent) AS nee_median,\n        quantile_cont(norepinephrine_equivalent, 0.99) AS nee_p99,\n        max(norepinephrine_equivalent) AS nee_max\n    FROM read_parquet('{sql_path(output)}')\n    GROUP BY drug\n)\nSELECT\n    r.drug,\n    r.raw_records,\n    c.clean_records,\n    r.raw_records - c.clean_records AS excluded_records,\n    round(100.0 * c.clean_records / r.raw_records, 3) AS retained_pct,\n    round(c.nee_p01, 6) AS nee_p01,\n    round(c.nee_median, 6) AS nee_median,\n    round(c.nee_p99, 6) AS nee_p99,\n    round(c.nee_max, 6) AS nee_max\nFROM raw AS r\nLEFT JOIN clean AS c USING (drug)\nORDER BY raw_records DESC\n").fetchall()
print('STAGE 6 DOSE CLEANING COMPLETED')
for r in summary:
    print(f'{r[0]}: raw={r[1]}, clean={r[2]}, excluded={r[3]}, retained={r[4]}%, NEE p01={r[5]}, median={r[6]}, p99={r[7]}, max={r[8]}')
print(f'Output: {output}')
con.close()

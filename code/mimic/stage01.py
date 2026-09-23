import os
from pathlib import Path
import duckdb
root = Path(os.environ['MIMIC_RAW'])
derived = Path(os.environ['MIMIC_DERIVED'])
temp = Path(os.environ['REPRO_TEMP'])
derived.mkdir(parents=True, exist_ok=True)
temp.mkdir(parents=True, exist_ok=True)
inputevents = root / 'icu' / 'inputevents.csv.gz'
d_items = root / 'icu' / 'd_items.csv.gz'
output = derived / 'vasopressor_inputevents_stage1.parquet'
summary_csv = derived / 'vasopressor_stage1_summary.csv'
if output.exists():
    raise SystemExit(f'输出文件已存在，未覆盖：{output}')

def sql_path(path):
    return str(path).replace('\\', '/').replace("'", "''")
itemids = '229709,229764,221662,221289,229617,221906,221749,229632,229631,229630,222315'
con = duckdb.connect()
con.execute('SET threads = 1')
con.execute("SET memory_limit = '8GB'")
con.execute(f"SET temp_directory = '{sql_path(temp)}'")
con.execute('SET preserve_insertion_order = false')
extract_sql = f"\nCOPY (\n    SELECT\n        i.source_row_number,\n        i.subject_id,\n        i.hadm_id,\n        i.stay_id,\n        i.starttime,\n        i.endtime,\n        i.storetime,\n        i.itemid,\n        d.label AS item_label,\n        CASE\n            WHEN i.itemid IN (229709,229764) THEN 'angiotensin_ii'\n            WHEN i.itemid = 221662 THEN 'dopamine'\n            WHEN i.itemid IN (221289,229617) THEN 'epinephrine'\n            WHEN i.itemid = 221906 THEN 'norepinephrine'\n            WHEN i.itemid IN (221749,229632,229631,229630) THEN 'phenylephrine'\n            WHEN i.itemid = 222315 THEN 'vasopressin'\n        END AS drug,\n        i.amount,\n        i.amountuom,\n        i.rate,\n        i.rateuom,\n        i.patientweight,\n        i.orderid,\n        i.linkorderid,\n        i.statusdescription,\n        i.originalamount,\n        i.originalrate\n    FROM (SELECT ordinality AS source_row_number, * EXCLUDE (ordinality) FROM read_csv_auto('{sql_path(inputevents)}', header=true, sample_size=1000000, parallel=false) WITH ORDINALITY) AS i\n    INNER JOIN read_csv_auto('{sql_path(d_items)}', header=true) AS d\n        USING (itemid)\n    WHERE i.itemid IN ({itemids})\n    ORDER BY i.source_row_number\n)\nTO '{sql_path(output)}'\n(FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 250000)\n"
print('正在提取血管活性药事件，可能需要几分钟……')
con.execute(extract_sql)
summary_query = f"\nSELECT\n    drug,\n    itemid,\n    item_label,\n    count(*) AS records,\n    count(DISTINCT stay_id) AS icu_stays,\n    count(DISTINCT subject_id) AS patients,\n    min(starttime) AS first_start,\n    max(endtime) AS last_end,\n    count(*) FILTER (WHERE rate IS NULL) AS missing_rate,\n    count(*) FILTER (WHERE starttime IS NULL OR endtime IS NULL) AS missing_time,\n    count(*) FILTER (WHERE endtime < starttime) AS negative_duration\nFROM read_parquet('{sql_path(output)}')\nGROUP BY drug, itemid, item_label\nORDER BY icu_stays DESC, records DESC\n"
summary = con.execute(summary_query).fetchdf()
con.execute(f"COPY ({summary_query}) TO '{sql_path(summary_csv)}' (HEADER, DELIMITER ',')")
overall = con.execute(f"\nSELECT\n    count(*) AS records,\n    count(DISTINCT stay_id) AS icu_stays,\n    count(DISTINCT subject_id) AS patients,\n    min(starttime) AS first_start,\n    max(endtime) AS last_end,\n    count(*) FILTER (WHERE rate IS NULL) AS missing_rate,\n    count(*) FILTER (WHERE starttime IS NULL OR endtime IS NULL) AS missing_time,\n    count(*) FILTER (WHERE endtime < starttime) AS negative_duration\nFROM read_parquet('{sql_path(output)}')\n").fetchdf()
print('\nMIMIC-IV VASOPRESSOR STAGE 1 COMPLETED')
print('\nOVERALL')
print(overall.to_string(index=False))
print('\nBY ITEM')
print(summary.to_string(index=False))
print(f'\nParquet: {output}')
print(f'Summary: {summary_csv}')
con.close()

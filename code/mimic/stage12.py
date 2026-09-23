import os
from pathlib import Path
import duckdb
derived = Path(os.environ['MIMIC_DERIVED'])
cohort = derived / 'vasopressor_liberation_primary_cohort_stage3.parquet'
code_status = derived / 'preliberation_code_status_stage11.parquet'
output = derived / 'vasopressor_liberation_code_status_stage12.parquet'

def sql_path(path):
    return str(path).replace('\\', '/').replace("'", "''")
con = duckdb.connect()
con.execute('PRAGMA threads=1')
con.execute("PRAGMA memory_limit='8GB'")
con.execute("SET temp_directory = ?", [os.environ["REPRO_TEMP"]])
con.execute(f"\nCREATE OR REPLACE TEMP TABLE stage12 AS\nWITH code AS (\n    SELECT *\n    FROM read_parquet('{sql_path(code_status)}')\n),\nlatest_pre AS (\n    SELECT stay_id, value AS latest_pre_code_status\n    FROM (\n        SELECT\n            stay_id,\n            value,\n            row_number() OVER (\n                PARTITION BY stay_id\n                ORDER BY minutes_before_stop ASC, charttime DESC, itemid, source_row_number\n            ) AS rn\n        FROM code\n        WHERE minutes_before_stop >= 0\n    )\n    WHERE rn = 1\n),\nflags AS (\n    SELECT\n        stay_id,\n        max(CASE\n            WHEN lower(trim(value)) = 'comfort measures only'\n             AND minutes_before_stop >= 0\n            THEN 1 ELSE 0 END\n        ) AS any_cmo_before_stop,\n\n        max(CASE\n            WHEN lower(trim(value)) = 'comfort measures only'\n             AND minutes_before_stop < 0\n             AND minutes_before_stop >= -60\n            THEN 1 ELSE 0 END\n        ) AS cmo_within_1h_after_stop,\n\n        max(CASE\n            WHEN minutes_before_stop >= 0\n             AND (\n                 lower(value) LIKE '%dnr%'\n                 OR lower(value) LIKE '%dnar%'\n                 OR lower(value) LIKE '%dni%'\n                 OR lower(value) LIKE '%do not resuscitate%'\n                 OR lower(value) LIKE '%do not attempt resuscitation%'\n             )\n            THEN 1 ELSE 0 END\n        ) AS any_dnr_dni_before_stop\n    FROM code\n    GROUP BY stay_id\n)\nSELECT\n    c.*,\n    lp.latest_pre_code_status,\n    CASE WHEN lp.stay_id IS NOT NULL THEN 1 ELSE 0 END\n        AS code_status_documented_before_stop,\n\n    CASE\n        WHEN lower(trim(coalesce(lp.latest_pre_code_status, '')))\n             = 'comfort measures only'\n        THEN 1 ELSE 0\n    END AS cmo_active_at_stop,\n\n    coalesce(f.any_cmo_before_stop, 0) AS any_cmo_before_stop,\n    coalesce(f.cmo_within_1h_after_stop, 0) AS cmo_within_1h_after_stop,\n    coalesce(f.any_dnr_dni_before_stop, 0) AS any_dnr_dni_before_stop,\n\n    CASE\n        WHEN regexp_matches(\n            lower(coalesce(lp.latest_pre_code_status, '')),\n            'dnr|dnar|dni|do not resuscitate|do not attempt resuscitation'\n        )\n        THEN 1 ELSE 0\n    END AS dnr_dni_active_at_stop\nFROM read_parquet('{sql_path(cohort)}') c\nLEFT JOIN latest_pre lp USING (stay_id)\nLEFT JOIN flags f USING (stay_id)\n")
con.execute(f"\nCOPY stage12 TO '{sql_path(output)}'\n(FORMAT PARQUET, COMPRESSION ZSTD)\n")
print('STAGE 12 cohort written; historical reporting-only repair is retained in provenance.')
con.close()

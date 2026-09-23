import os
from pathlib import Path
import duckdb
derived = Path(os.environ['MIMIC_DERIVED'])
source = derived / 'stage17_cross_icu_restart_audit.parquet'
output = derived / 'stage18_competing_outcomes_cross_icu_corrected.parquet'

def sql_path(path):
    return str(path).replace('\\', '/').replace("'", "''")
con = duckdb.connect()
con.execute(f"\nCOPY (\n    SELECT\n        * EXCLUDE (\n            restart_24h,\n            next_episode_start,\n            failure_24h,\n            both_events_24h,\n            simultaneous_events,\n            competing_outcome,\n            competing_outcome_code,\n            time_to_first_event_hours,\n            time_to_restart_hours,\n            revised_restart_24h,\n            revised_restart_time,\n            revised_competing_outcome\n        ),\n\n        revised_restart_24h AS restart_24h,\n        revised_restart_time AS next_episode_start,\n\n        CASE\n            WHEN revised_restart_24h = 1\n              OR death_24h = 1\n            THEN 1 ELSE 0\n        END AS failure_24h,\n\n        CASE\n            WHEN revised_restart_24h = 1\n             AND death_24h = 1\n            THEN 1 ELSE 0\n        END AS both_events_24h,\n\n        CASE\n            WHEN revised_restart_24h = 1\n             AND death_24h = 1\n             AND revised_restart_time = deathtime\n            THEN 1 ELSE 0\n        END AS simultaneous_events,\n\n        revised_competing_outcome AS competing_outcome,\n\n        CASE revised_competing_outcome\n            WHEN 'durable_liberation_alive' THEN 0\n            WHEN 'restart_first' THEN 1\n            WHEN 'death_first' THEN 2\n            ELSE 9\n        END AS competing_outcome_code,\n\n        CASE\n            WHEN revised_competing_outcome = 'restart_first'\n            THEN date_diff(\n                'second',\n                episode_end,\n                revised_restart_time\n            ) / 3600.0\n\n            WHEN revised_competing_outcome = 'death_first'\n            THEN date_diff(\n                'second',\n                episode_end,\n                deathtime\n            ) / 3600.0\n\n            ELSE NULL\n        END AS time_to_first_event_hours,\n\n        CASE\n            WHEN revised_restart_24h = 1\n            THEN date_diff(\n                'second',\n                episode_end,\n                revised_restart_time\n            ) / 3600.0\n            ELSE NULL\n        END AS time_to_restart_hours\n\n    FROM read_parquet('{sql_path(source)}')\n)\nTO '{sql_path(output)}'\n(FORMAT PARQUET, COMPRESSION ZSTD)\n")
qa = con.execute(f"\nSELECT\n    count(*) AS development_validation_stays,\n    sum(CASE WHEN competing_outcome_code = 0 THEN 1 ELSE 0 END)\n        AS durable_liberation,\n    sum(CASE WHEN competing_outcome_code = 1 THEN 1 ELSE 0 END)\n        AS restart_first,\n    sum(CASE WHEN competing_outcome_code = 2 THEN 1 ELSE 0 END)\n        AS death_first,\n    sum(CASE WHEN competing_outcome_code = 9 THEN 1 ELSE 0 END)\n        AS unclassifiable\nFROM read_parquet('{sql_path(output)}')\nWHERE anchor_year_group <> '2020 - 2022'\n").fetchdf()
print('\nSTAGE 18 CORRECTED OUTCOME COHORT COMPLETED')
print(qa.to_string(index=False))
print(f'\nOutput: {output}')
print('Locked-period outcomes were not displayed.')
con.close()

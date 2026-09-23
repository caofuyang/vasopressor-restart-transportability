import os
from pathlib import Path
import duckdb
events = Path(os.environ['MIMIC_DERIVED']) / 'vasopressor_inputevents_stage1.parquet'
root = Path(os.environ['MIMIC_RAW'])
output = Path(os.environ['MIMIC_DERIVED']) / 'vasopressor_liberation_attempts_stage2.parquet'
temp = Path(os.environ['REPRO_TEMP'])
if output.exists():
    raise SystemExit(f'输出已存在，未覆盖：{output}')

def sql_path(p):
    return str(p).replace('\\', '/').replace("'", "''")
con = duckdb.connect()
con.execute('SET threads = 1')
con.execute("SET memory_limit = '8GB'")
con.execute(f"SET temp_directory = '{sql_path(temp)}'")
con.execute('SET preserve_insertion_order = true')
con.execute(f"\nCREATE TEMP TABLE candidate_attempts AS\nWITH valid_events AS (\n    SELECT *\n    FROM read_parquet('{sql_path(events)}')\n    WHERE itemid <> 229617\n      AND rate IS NOT NULL\n      AND rate > 0\n      AND endtime > starttime\n      AND statusdescription IN\n          ('ChangeDose/Rate', 'FinishedRunning', 'Stopped', 'Paused')\n),\nordered AS (\n    SELECT *,\n        max(endtime) OVER (\n            PARTITION BY stay_id\n            ORDER BY starttime, endtime, source_row_number\n            ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING\n        ) AS prior_max_end\n    FROM valid_events\n),\nmarked AS (\n    SELECT *,\n        CASE\n            WHEN prior_max_end IS NULL\n              OR starttime > prior_max_end + INTERVAL '60 minutes'\n            THEN 1 ELSE 0\n        END AS new_episode\n    FROM ordered\n),\ngrouped AS (\n    SELECT *,\n        sum(new_episode) OVER (\n            PARTITION BY stay_id\n            ORDER BY starttime, endtime, source_row_number\n            ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW\n        ) AS episode_id\n    FROM marked\n),\nepisodes AS (\n    SELECT\n        subject_id,\n        hadm_id,\n        stay_id,\n        episode_id,\n        min(starttime) AS episode_start,\n        max(endtime) AS episode_end,\n        arg_max(statusdescription, struct_pack(event_end := endtime, reverse_row := -source_row_number)) AS terminal_status,\n        arg_max(drug, struct_pack(event_end := endtime, reverse_row := -source_row_number)) AS terminal_drug,\n        count(*) AS source_records,\n        count(DISTINCT drug) AS distinct_drugs,\n        date_diff('minute', min(starttime), max(endtime)) / 60.0\n            AS duration_hours\n    FROM grouped\n    GROUP BY subject_id, hadm_id, stay_id, episode_id\n),\nsequenced AS (\n    SELECT *,\n        lead(episode_start) OVER (\n            PARTITION BY stay_id\n            ORDER BY episode_start, episode_end\n        ) AS next_episode_start\n    FROM episodes\n),\neligible AS (\n    SELECT\n        s.*,\n        i.intime,\n        i.outtime,\n        a.deathtime,\n        CASE\n            WHEN s.next_episode_start\n                 <= s.episode_end + INTERVAL '24 hours'\n            THEN 1 ELSE 0\n        END AS restart_24h,\n        CASE\n            WHEN a.deathtime > s.episode_end\n             AND a.deathtime <= s.episode_end + INTERVAL '24 hours'\n            THEN 1 ELSE 0\n        END AS death_24h\n    FROM sequenced AS s\n    INNER JOIN read_csv_auto(\n        '{sql_path(root / 'icu' / 'icustays.csv.gz')}',\n        header=true, sample_size=1000000\n    ) AS i\n        ON s.stay_id = i.stay_id\n    INNER JOIN read_csv_auto(\n        '{sql_path(root / 'hosp' / 'admissions.csv.gz')}',\n        header=true, sample_size=1000000\n    ) AS a\n        ON s.subject_id = a.subject_id\n       AND s.hadm_id = a.hadm_id\n    WHERE s.duration_hours >= 6\n      AND s.terminal_status IN ('Stopped', 'FinishedRunning')\n      AND s.episode_end <= i.outtime + INTERVAL '1 hour'\n),\nnumbered AS (\n    SELECT *,\n        row_number() OVER (\n            PARTITION BY stay_id\n            ORDER BY episode_end\n        ) AS candidate_number\n    FROM eligible\n)\nSELECT *,\n    CASE\n        WHEN restart_24h = 1 OR death_24h = 1 THEN 1 ELSE 0\n    END AS failure_24h,\n    CASE\n        WHEN outtime >= episode_end + INTERVAL '24 hours'\n          OR restart_24h = 1\n          OR death_24h = 1\n        THEN 1 ELSE 0\n    END AS strict_24h_observable,\n    CASE\n        WHEN outtime > episode_end\n         AND outtime < episode_end + INTERVAL '24 hours'\n         AND restart_24h = 0\n         AND death_24h = 0\n        THEN 1 ELSE 0\n    END AS early_alive_icu_exit\nFROM numbered\n")
con.execute(f"\nCOPY candidate_attempts\nTO '{sql_path(output)}'\n(FORMAT PARQUET, COMPRESSION ZSTD)\n")
terminal = con.execute('\nSELECT\n    terminal_status,\n    count(*) AS candidate_attempts,\n    count(DISTINCT stay_id) AS stays,\n    sum(restart_24h) AS restart_24h,\n    round(100.0 * avg(restart_24h), 2) AS restart_pct,\n    sum(death_24h) AS death_24h\nFROM candidate_attempts\nGROUP BY terminal_status\nORDER BY candidate_attempts DESC\n').fetchdf()
first_attempt = con.execute('\nSELECT\n    count(*) AS first_attempts,\n    sum(restart_24h) AS restart_24h,\n    sum(death_24h) AS death_24h,\n    sum(failure_24h) AS composite_failure_24h,\n    round(100.0 * avg(failure_24h), 2) AS composite_failure_pct,\n    sum(strict_24h_observable) AS strict_observable,\n    sum(early_alive_icu_exit) AS early_alive_icu_exit\nFROM candidate_attempts\nWHERE candidate_number = 1\n').fetchdf()
strict = con.execute('\nSELECT\n    count(*) AS strict_first_attempts,\n    sum(failure_24h) AS failures,\n    round(100.0 * avg(failure_24h), 2) AS failure_pct\nFROM candidate_attempts\nWHERE candidate_number = 1\n  AND strict_24h_observable = 1\n').fetchdf()
print('STAGE 2 COMPLETED')
print('\nBY TERMINAL STATUS')
print(terminal.to_string(index=False))
print('\nFIRST ATTEMPT - PRAGMATIC')
print(first_attempt.to_string(index=False))
print('\nFIRST ATTEMPT - STRICT 24H OBSERVATION')
print(strict.to_string(index=False))
print(f'\nOutput: {output}')
con.close()

"""Build the independent post-audit corrected cohort from frozen rules."""

from __future__ import annotations

from pathlib import Path

import duckdb
import os


HERE = Path(__file__).resolve().parent
RESULTS = Path(os.environ["MIMIC_RUN"]) / "post_audit"
SOURCE = Path(os.environ["MIMIC_DERIVED"]) / "stage18_competing_outcomes_cross_icu_corrected.parquet"
ADMISSIONS = Path(os.environ["MIMIC_RAW"]) / "hosp/admissions.csv.gz"
OUTPUT = RESULTS / "stage18_post_audit_corrected_cohort.parquet"

TRAINING_GROUPS = (
    "2008 - 2010",
    "2011 - 2013",
    "2014 - 2016",
    "2017 - 2019",
)
TEST_GROUP = "2020 - 2022"


def sql_path(path: Path) -> str:
    return str(path).replace("\\", "/").replace("'", "''")


def main() -> None:
    for path in (SOURCE, ADMISSIONS):
        if not path.is_file():
            raise FileNotFoundError(path)
    RESULTS.mkdir(parents=True, exist_ok=True)
    if OUTPUT.exists():
        raise SystemExit(f"Refusing to overwrite existing corrected cohort: {OUTPUT}")

    source = sql_path(SOURCE)
    admissions = sql_path(ADMISSIONS)
    output = sql_path(OUTPUT)
    training_sql = ", ".join(f"'{group}'" for group in TRAINING_GROUPS)

    con = duckdb.connect(":memory:")
    con.execute("SET threads = 1")
    con.execute("SET preserve_insertion_order = true")

    con.execute(
        f"""
        CREATE TEMP TABLE corrected AS
        WITH source AS (
            SELECT
                s.*,
                a.dischtime AS raw_dischtime,
                a.deathtime AS raw_deathtime,
                a.hospital_expire_flag AS raw_hospital_expire_flag,
                CASE
                    WHEN s.anchor_year_group IN ({training_sql}) THEN 'development'
                    WHEN s.anchor_year_group = '{TEST_GROUP}' THEN 'temporal_validation'
                    ELSE 'unmapped'
                END AS corrected_analysis_split
            FROM read_parquet('{source}') s
            INNER JOIN read_csv_auto('{admissions}', header=true) a
              ON s.hadm_id = a.hadm_id
             AND s.subject_id = a.subject_id
        ), flags AS (
            SELECT
                *,
                CASE
                    WHEN next_episode_start > episode_end
                     AND next_episode_start <= episode_end + INTERVAL '24 hours'
                    THEN 1 ELSE 0
                END AS corrected_restart_in_24h,
                CASE
                    WHEN raw_deathtime > episode_end
                     AND raw_deathtime <= episode_end + INTERVAL '24 hours'
                    THEN 1 ELSE 0
                END AS corrected_death_in_24h,
                CASE
                    WHEN next_episode_start IS NOT NULL
                     AND next_episode_start <= episode_end
                    THEN 1 ELSE 0
                END AS restart_time_at_or_before_origin,
                CASE
                    WHEN raw_deathtime < episode_end THEN 'invalid_death_before_origin'
                    WHEN raw_deathtime = episode_end THEN 'invalid_death_at_origin'
                    WHEN raw_hospital_expire_flag = 1 AND raw_deathtime IS NULL
                    THEN 'indeterminate_missing_death_time'
                    WHEN raw_hospital_expire_flag = 0 AND raw_dischtime < episode_end
                    THEN 'invalid_alive_discharge_before_origin'
                    ELSE 'valid_origin'
                END AS origin_status
            FROM source
        ), classified AS (
            SELECT
                *,
                CASE
                    WHEN origin_status <> 'valid_origin' THEN origin_status
                    WHEN corrected_restart_in_24h = 1
                     AND corrected_death_in_24h = 1
                     AND next_episode_start <= raw_deathtime
                    THEN 'restart_first'
                    WHEN corrected_restart_in_24h = 1
                     AND corrected_death_in_24h = 1
                     AND raw_deathtime < next_episode_start
                    THEN 'death_first'
                    WHEN corrected_restart_in_24h = 1 THEN 'restart_first'
                    WHEN corrected_death_in_24h = 1 THEN 'death_first'
                    WHEN outtime >= episode_end + INTERVAL '24 hours'
                    THEN 'durable_liberation_alive_confirmed'
                    ELSE 'indeterminate_incomplete_24h_observation'
                END AS corrected_status
            FROM flags
        )
        SELECT
            *,
            CASE corrected_status
                WHEN 'durable_liberation_alive_confirmed' THEN 0
                WHEN 'restart_first' THEN 1
                WHEN 'death_first' THEN 2
                ELSE NULL
            END AS corrected_outcome_code,
            CASE
                WHEN corrected_status IN (
                    'durable_liberation_alive_confirmed',
                    'restart_first',
                    'death_first'
                ) THEN 1 ELSE 0
            END AS primary_analysis_include,
            CASE
                WHEN corrected_status IN (
                    'durable_liberation_alive_confirmed',
                    'restart_first',
                    'death_first'
                )
                 AND episode_end <= outtime
                 AND episode_end <= raw_dischtime
                THEN 1 ELSE 0
            END AS strict_sensitivity_include
        FROM classified
        """
    )

    con.execute(
        f"COPY corrected TO '{output}' (FORMAT PARQUET, COMPRESSION ZSTD)"
    )

    flow = con.execute(
        """
        SELECT
            corrected_analysis_split AS analysis_split,
            corrected_status,
            count(*) AS records,
            count(DISTINCT subject_id) AS unique_subjects
        FROM corrected
        GROUP BY corrected_analysis_split, corrected_status
        ORDER BY corrected_analysis_split, corrected_status
        """
    ).fetchdf()
    flow.to_csv(RESULTS / "corrected_cohort_flow.csv", index=False)

    outcomes = con.execute(
        """
        SELECT
            corrected_analysis_split AS analysis_split,
            corrected_status,
            count(*) AS records,
            count(DISTINCT subject_id) AS unique_subjects,
            round(
                100.0 * count(*) / sum(count(*)) OVER (
                    PARTITION BY corrected_analysis_split
                ), 2
            ) AS percent_within_included_split
        FROM corrected
        WHERE primary_analysis_include = 1
        GROUP BY corrected_analysis_split, corrected_status
        ORDER BY corrected_analysis_split,
            CASE corrected_status
                WHEN 'durable_liberation_alive_confirmed' THEN 0
                WHEN 'restart_first' THEN 1
                WHEN 'death_first' THEN 2
                ELSE 9
            END
        """
    ).fetchdf()
    outcomes.to_csv(RESULTS / "corrected_primary_outcomes_by_split.csv", index=False)

    strict = con.execute(
        """
        SELECT
            corrected_analysis_split AS analysis_split,
            corrected_status,
            count(*) AS records,
            count(DISTINCT subject_id) AS unique_subjects,
            round(
                100.0 * count(*) / sum(count(*)) OVER (
                    PARTITION BY corrected_analysis_split
                ), 2
            ) AS percent_within_strict_split
        FROM corrected
        WHERE strict_sensitivity_include = 1
        GROUP BY corrected_analysis_split, corrected_status
        ORDER BY corrected_analysis_split,
            CASE corrected_status
                WHEN 'durable_liberation_alive_confirmed' THEN 0
                WHEN 'restart_first' THEN 1
                WHEN 'death_first' THEN 2
                ELSE 9
            END
        """
    ).fetchdf()
    strict.to_csv(RESULTS / "corrected_strict_outcomes_by_split.csv", index=False)

    qa = con.execute(
        """
        SELECT
            count(*) AS source_records,
            count(DISTINCT stay_id) AS unique_stays,
            sum(primary_analysis_include) AS primary_included,
            sum(strict_sensitivity_include) AS strict_included,
            sum(restart_time_at_or_before_origin) AS restart_at_or_before_origin,
            count(*) FILTER (WHERE corrected_analysis_split = 'unmapped') AS unmapped,
            count(*) FILTER (
                WHERE corrected_status = 'invalid_death_before_origin'
            ) AS death_before_origin,
            count(*) FILTER (
                WHERE corrected_status = 'invalid_death_at_origin'
            ) AS death_at_origin,
            count(*) FILTER (
                WHERE corrected_status = 'indeterminate_missing_death_time'
            ) AS missing_death_time,
            count(*) FILTER (
                WHERE corrected_status = 'invalid_alive_discharge_before_origin'
            ) AS alive_discharge_before_origin,
            count(*) FILTER (
                WHERE corrected_status = 'indeterminate_incomplete_24h_observation'
            ) AS incomplete_observation
        FROM corrected
        """
    ).fetchdf()
    qa.to_csv(RESULTS / "corrected_cohort_QA.csv", index=False)

    print("POST-AUDIT CORRECTED COHORT CREATED")
    print("\nQA")
    print(qa.to_string(index=False))
    print("\nFLOW")
    print(flow.to_string(index=False))
    print("\nPRIMARY CONFIRMED OUTCOMES")
    print(outcomes.to_string(index=False))
    print("\nSTRICT SENSITIVITY OUTCOMES")
    print(strict.to_string(index=False))
    print(f"\nOutput: {OUTPUT}")
    con.close()


if __name__ == "__main__":
    main()

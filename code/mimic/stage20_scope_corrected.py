"""Apply the frozen same-hospitalization, strictly-after restart scope correction.

This is a post-audit correction. It writes a new cohort and aggregate QA only;
the prior corrected cohort is read-only and is never overwritten.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import duckdb
import os


HERE = Path(__file__).resolve().parent
OUT = Path(os.environ["MIMIC_RUN"]) / "scope_corrected"
PRIVATE = OUT / "restricted"
SOURCE = Path(os.environ["MIMIC_RUN"]) / "post_audit/stage18_post_audit_corrected_cohort.parquet"
STAGE13 = Path(os.environ["MIMIC_DERIVED"]) / "vasopressor_liberation_competing_outcomes_stage13.parquet"
VASO = Path(os.environ["MIMIC_DERIVED"]) / "vasopressor_inputevents_nee_clean_stage6.parquet"
RULES = HERE / "00_MIMIC_RESTART_SCOPE_CORRECTION_FROZEN.md"
MANIFEST = HERE / "mimic_restart_scope_correction_manifest.json"
LOCK = HERE / "MIMIC_RESTART_SCOPE_RULES_LOCK.sha256"


def sql_path(path: Path) -> str:
    return str(path).replace("\\", "/").replace("'", "''")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    for path in (SOURCE, STAGE13, VASO, RULES, MANIFEST, LOCK):
        if not path.is_file():
            raise FileNotFoundError(path)
    if OUT.exists():
        raise SystemExit(f"Refusing to overwrite: {OUT}")
    PRIVATE.mkdir(parents=True)

    frozen = json.loads(MANIFEST.read_text(encoding="utf-8"))
    expected = frozen["rules"]["sha256"]
    if sha256(RULES) != expected:
        raise RuntimeError("Frozen rules hash mismatch")

    con = duckdb.connect(":memory:")
    con.execute("SET threads=1")
    con.execute(f"CREATE VIEW source AS SELECT * FROM read_parquet('{sql_path(SOURCE)}')")
    con.execute(f"CREATE VIEW s13 AS SELECT * FROM read_parquet('{sql_path(STAGE13)}')")
    con.execute(f"CREATE VIEW vaso AS SELECT * FROM read_parquet('{sql_path(VASO)}')")
    con.execute(
        """
        CREATE TEMP TABLE corrected AS
        WITH cross_candidates AS (
            SELECT
                s.stay_id AS index_stay_id,
                min(v.starttime) AS first_same_hadm_cross_icu_restart
            FROM source s
            INNER JOIN vaso v
              ON s.subject_id = v.subject_id
             AND s.hadm_id = v.hadm_id
             AND s.stay_id <> v.stay_id
             AND v.starttime > s.episode_end
             AND v.starttime <= s.episode_end + INTERVAL '24 hours'
            GROUP BY s.stay_id
        ), restart_times AS (
            SELECT
                s.*,
                b.next_episode_start AS initial_within_stay_restart,
                x.first_same_hadm_cross_icu_restart,
                CASE
                    WHEN b.next_episode_start > s.episode_end
                     AND b.next_episode_start <= s.episode_end + INTERVAL '24 hours'
                     AND x.first_same_hadm_cross_icu_restart IS NOT NULL
                    THEN least(b.next_episode_start, x.first_same_hadm_cross_icu_restart)
                    WHEN b.next_episode_start > s.episode_end
                     AND b.next_episode_start <= s.episode_end + INTERVAL '24 hours'
                    THEN b.next_episode_start
                    ELSE x.first_same_hadm_cross_icu_restart
                END AS scoped_restart_time
            FROM source s
            LEFT JOIN s13 b USING (stay_id)
            LEFT JOIN cross_candidates x ON x.index_stay_id = s.stay_id
        ), reclassified AS (
            SELECT
                *,
                CASE
                    WHEN origin_status <> 'valid_origin' THEN origin_status
                    WHEN scoped_restart_time IS NOT NULL
                     AND corrected_death_in_24h = 1
                     AND scoped_restart_time <= raw_deathtime THEN 'restart_first'
                    WHEN scoped_restart_time IS NOT NULL
                     AND corrected_death_in_24h = 1
                     AND raw_deathtime < scoped_restart_time THEN 'death_first'
                    WHEN scoped_restart_time IS NOT NULL THEN 'restart_first'
                    WHEN corrected_death_in_24h = 1 THEN 'death_first'
                    WHEN outtime >= episode_end + INTERVAL '24 hours'
                    THEN 'durable_liberation_alive_confirmed'
                    ELSE 'indeterminate_incomplete_24h_observation'
                END AS scope_corrected_status
            FROM restart_times
        )
        SELECT
            corrected_status AS prior_corrected_status,
            * EXCLUDE(
                corrected_restart_in_24h, corrected_status,
                corrected_outcome_code, primary_analysis_include,
                strict_sensitivity_include
            ),
            CASE WHEN scoped_restart_time IS NOT NULL THEN 1 ELSE 0 END AS corrected_restart_in_24h,
            scope_corrected_status AS corrected_status,
            CASE scope_corrected_status
                WHEN 'durable_liberation_alive_confirmed' THEN 0
                WHEN 'restart_first' THEN 1
                WHEN 'death_first' THEN 2
                ELSE NULL
            END AS corrected_outcome_code,
            CASE WHEN scope_corrected_status IN (
                'durable_liberation_alive_confirmed','restart_first','death_first'
            ) THEN 1 ELSE 0 END AS primary_analysis_include,
            CASE WHEN scope_corrected_status IN (
                'durable_liberation_alive_confirmed','restart_first','death_first'
            ) AND episode_end <= outtime AND episode_end <= raw_dischtime
            THEN 1 ELSE 0 END AS strict_sensitivity_include
        FROM reclassified
        """
    )
    cohort = PRIVATE / "stage18_post_audit_scope_corrected_cohort.parquet"
    con.execute(f"COPY corrected TO '{sql_path(cohort)}' (FORMAT PARQUET, COMPRESSION ZSTD)")

    con.execute(
        f"""
        COPY (
          SELECT corrected_analysis_split, corrected_status, count(*) records,
                 count(DISTINCT subject_id) unique_subjects
          FROM corrected GROUP BY ALL ORDER BY 1,2
        ) TO '{sql_path(OUT / 'cohort_flow.csv')}' (HEADER, DELIMITER ',')
        """
    )
    con.execute(
        f"""
        COPY (
          SELECT corrected_analysis_split, corrected_status, count(*) records
          FROM corrected WHERE primary_analysis_include=1
          GROUP BY ALL ORDER BY 1,2
        ) TO '{sql_path(OUT / 'primary_outcomes_by_split.csv')}' (HEADER, DELIMITER ',')
        """
    )
    con.execute(
        f"""
        COPY (
          SELECT corrected_analysis_split, prior_corrected_status AS prior_status,
                 corrected_status AS revised_status, count(*) records
          FROM corrected
          WHERE prior_corrected_status <> corrected_status
          GROUP BY ALL ORDER BY 1,2,3
        ) TO '{sql_path(OUT / 'label_change_matrix.csv')}' (HEADER, DELIMITER ',')
        """
    )
    qa = con.execute(
        """
        SELECT count(*) source_records, count(DISTINCT stay_id) unique_stays,
               sum(primary_analysis_include) primary_records,
               sum(strict_sensitivity_include) strict_records,
               count(*) FILTER (WHERE corrected_analysis_split='development' AND primary_analysis_include=1) development_primary,
               count(*) FILTER (WHERE corrected_analysis_split='temporal_validation' AND primary_analysis_include=1) validation_primary,
               count(*) FILTER (WHERE prior_corrected_status <> corrected_status) label_changes
        FROM corrected
        """
    ).fetchone()
    report = dict(zip([
        "source_records","unique_stays","primary_records","strict_records",
        "development_primary","validation_primary","label_changes"
    ], qa))
    report.update({
        "rules_hash_match": True,
        "rules_sha256": expected,
        "input_sha256": {str(p): sha256(p) for p in (SOURCE, STAGE13, VASO)},
        "patient_level_output": str(cohort),
        "patient_level_rows_in_public_outputs": 0,
    })
    (OUT / "cohort_QA.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    con.close()


if __name__ == "__main__":
    main()

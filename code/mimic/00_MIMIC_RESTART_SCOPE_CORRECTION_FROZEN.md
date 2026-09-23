# MIMIC-IV cross-ICU restart scope correction

## Status

This is a post-audit correction made after a discrepancy was identified between the manuscript description and the recovered Stage 17 implementation. It was not prospectively prespecified. The rule was frozen before any corrected model performance was calculated or viewed.

## Frozen rule

1. A cross-ICU restart must involve the same `subject_id` and the same `hadm_id` as the index episode, and a different `stay_id`.
2. A qualifying restart start time must be strictly later than `episode_end` and no later than 24 hours after `episode_end`.
3. A candidate exactly equal to `episode_end` is not counted as a restart. Candidate search continues through all later eligible starts rather than stopping at an equal-time minimum.
4. The original within-ICU restart search remains unchanged. The revised restart time is the earliest qualifying within-ICU or same-admission cross-ICU restart.
5. Death and restart ordering retains the frozen competing-outcome rule: restart first when the qualifying restart precedes or equals death; death first when death precedes restart. A restart exactly at the study origin is not an event after origin.
6. Records with no qualifying restart or death are included in the no-recorded-event category only when the previously frozen 24-hour observation requirement is met; otherwise they remain indeterminate.

## Observed audit impact before model updating

- One development record currently classified as `restart_first` used a selected cross-ICU start from a different `hadm_id`.
- No temporal-validation record used a selected cross-ICU start from a different `hadm_id`.
- No record had an equal-origin cross-ICU candidate that masked a later strictly-after same-admission candidate.
- Applying the frozen scope rule changes the one affected development record to `indeterminate_incomplete_24h_observation`.

## Prohibitions

- The rule, feature set, temporal split, model specifications, hyperparameters, and bootstrap units must not be changed in response to corrected performance.
- Historical files are retained and must not be overwritten.
- Patient identifiers must not be exported in public or aggregate deliverables.


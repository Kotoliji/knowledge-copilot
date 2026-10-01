# Engineering On-Call Runbook

## On-call rotation

Each product team runs a weekly on-call rotation. The rotation switches every Monday at 10:00 UTC. The on-call engineer must be reachable within 15 minutes during the shift and keeps the PagerDuty app installed with notifications enabled.

## Severity levels

- SEV1: full outage or data loss affecting all customers. Response time: 15 minutes. Status page update required within 30 minutes.
- SEV2: major feature unavailable or severe degradation for many customers. Response time: 30 minutes.
- SEV3: minor degradation or a bug with a workaround. Response time: next business day.

## Escalation process

If the on-call engineer does not acknowledge a page within 15 minutes, PagerDuty escalates to the secondary on-call. After another 15 minutes without acknowledgement, the incident escalates to the engineering manager of the team. SEV1 incidents additionally page the Incident Commander on duty immediately.

## Incident communication

For SEV1 and SEV2 incidents, open a dedicated Slack channel named `#inc-<date>-<short-name>` and post updates at least every 30 minutes. Customer-facing updates go to the status page and are written by the Incident Commander, never by the on-call engineer directly.

## Post-incident review

Every SEV1 and SEV2 incident requires a blameless post-incident review document within 5 working days. The review must include a timeline, root cause, customer impact, and at least one action item with an owner and a due date.

## Compensation

On-call shifts are compensated with a flat allowance of 150 EUR per week plus 1.5x hourly rate for time actively spent on incidents outside working hours.

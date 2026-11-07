# Production Paperless worker

This separate Compose project pulls OCR and metadata operations from
`https://paperless.atlas.home.arpa`. The Atlas web deployment uses Paperless Nano;
the old webhook and direct-writeback AI services stay stopped.

Worker image:
`ghcr.io/enucatl/paperless-nano-worker@sha256:0f54a54c5e158476f47c9b828c5fd66b0535fafa95659fe90627cca5e920beef`
(`v3.4.2-nano-97bc71617`).

The private `.env` selects this image, production URL,
`http://cerebrum.home.arpa:8108` OCR endpoint, `inception/mercury-2.5` metadata
model and absolute paths to the two secret files. Secrets are mode 600, owned
by UID 1000. The worker account has only parser and metadata capabilities.

For a new checkout, copy `.env.example` to `.env` and provision the two private
secret files before running the weekly workflow. The example uses paths relative
to this directory; the current installation uses absolute paths.

The weekly `vllm-weekly-window.timer` on Proxmox is the normal entry point.
It waits for healthy extraction services, runs temporary `workstation --once`
containers serially for its configured hour, then stops the model services it
started. A running pass finishes before cleanup. The resident worker and extraction
services use `restart: "no"` and stay stopped when Cerebrum boots.

The timer runs Tuesdays at 01:05 UTC (02:05 CET in winter). Pending work waits on
Atlas between weekly windows. Do not start a resident workstation container: it
would compete with the scheduled worker's private state lock.

Keep the `paperless-workstation_workstation_state` volume: it holds computed
results pending delivery. The host CA bundle is mounted read-only and selected
through `REQUESTS_CA_BUNDLE`; HTTPS certificate verification stays enabled.
Atlas exposes `/api/operations/` using Paperless token authentication, while the
main page retains interactive Authelia authentication.

Start the existing Paddle extraction services before processing PDFs. This
worker accepts PDFs only and needs no server database, Redis, media mounts or
inbound port. Model deployment configuration remains in the parent project.

Private rollback backups and migration evidence are under
`backups/nano-migration-20261107/`; matching originals and the deployment runbook
are at Atlas `/opt/docker/paperless-ai/tmp/nano-migration-20261107/` and
`/opt/docker/paperless-ai/docs/nano-deployment.md`.

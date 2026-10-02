# Condition Desk

Train condition monitoring for the depot. Four subsystems, one interface.

Built for NEBULA X ("The Living Railway"), Problem Statement 3.

A maintenance planner picks a subsystem, drops in a data file, and gets back a maintenance record:
the finding, the evidence the model actually read, and what to do next. Results download as the
submission-schema CSV.

## Context

NEBULA X was a Singapore rail hackathon run by LTA with NUS, Google Cloud, SMRT and SBS Transit.
I competed in team "cooked coders". This repo is my own build of Problem Statement 3: the models,
the validation work in `MODEL_REPORT.md`, and the app. The team's graded entry was a teammate's
repo ([theejer/lta-nebulax-submission](https://github.com/theejer/lta-nebulax-submission)), so
the official scores belong to that entry, not to this code.

## What it does

| Subsystem | Input | Answer | Validated score |
|---|---|---|---|
| Structural Health | Single-column bogie stress recording | Cumulative fatigue damage | 0.973 (LOO MAPE 2.71%) |
| Air Conditioning | Per-car temperature workbook (.xlsx) | All 8 cars ranked by leak likelihood | 0.979 over 6 cases |
| Doors | Continuous door motor stream | Every open/close cycle labelled | Not measurable, see below |
| Rail Corrugation | 1 s, 10 kHz axle-box recording | Normal, Side I or Side II | macro F1 0.745 +/- 0.027 |

Methods, benchmarks, rejected alternatives and honest uncertainty are in
[`MODEL_REPORT.md`](MODEL_REPORT.md). Three of those four numbers are optimistic for different
structural reasons and the report says so rather than quoting them as solid.

## Two things worth knowing

**Rail has a speed confound, and we measured it.** Every one of the 133 training files below
9.7 m/s is Normal, and all 38 corrugated files are above it, so "slow means Normal" is free
accuracy on half the training set. The held-out files have a near-identical speed distribution, so
it transfers, but the real task is only the 139 files moving fast enough to be corrugated.

**Door's cross-validation is blind, not passing.** The training data is perfectly separable, so
every model scores 1.000 and cross-validation can rank nothing. Rather than trust that, we built
three independent alternatives (a bare threshold, stream-relative features, per-operation
thresholds) and they agree with the shipped model on all 38 test cycles. The app marks the seven
cycles that fall in a range containing none of the 110 training cycles, because those are the real
uncertainty.

## Running it

Needs Python 3.11+ and Node 22+.

```bash
# interface
cd app/frontend && npm install && npm run build
cp -r out ../backend/static

# api, serving both
cd ../backend && pip install -r requirements.txt
uvicorn main:app --port 8077
```

Then open http://127.0.0.1:8077.

### Deploying

One container serves the API and the built interface, so there is no CORS layer and one thing to
deploy. No local Docker is needed; Cloud Build builds from source.

```bash
gcloud run deploy condition-desk --source . --region asia-southeast1 \
  --allow-unauthenticated --memory 2Gi --timeout 300
```

## Layout

```
app/backend/     FastAPI: /api/predict/<subsystem>, /api/export/<subsystem>
app/backend/models/   frozen models (17 KB total; nothing trains at request time)
app/frontend/    Next.js, static export
src/             the four pipelines, single source of truth for the model code
src/experiments/ validation harness and every benchmark in MODEL_REPORT.md
predictions/     this app's predictions for the held-out test files
```

## Data

The training and test data come from the organisers' repo,
[aochinwen/NebulaX-Hackathon-ProblemStatement](https://github.com/aochinwen/NebulaX-Hackathon-ProblemStatement),
and are not copied here (rail alone is 6 GB). The scripts in `src/experiments/` read two
environment variables: `NEBULAX_DATA` for the data folder (`ACV/`, `SHM/`, `Door/`, plus cached
features) and `NEBULAX_LABELS` for the training labels. Both default to `./data` and `./data/labels`.

`src/` is not duplicated into the backend. The container copies it, and `app/backend/service.py`
imports from it, so the app and the reported results cannot drift apart.

## Notes on the models

- **SHM** is a calculation, not a fit: rainflow counting (ASTM E1049-85) plus Miner's rule, with
  the S-N exponent pinned to 5 on physical grounds and one constant fitted over 64 files.
- **Rail** uses two feature banks. Frequency bands are what the sensor measures; wavelength bands
  come from resampling each channel at constant distance via the tacho, which makes them
  speed-invariant. The union beat frequency alone on 16 of 20 paired cross-validation seeds.
- **Door** ships a single-feature threshold. It matches a 33-feature gradient boosting model on all
  38 test cycles, and a threshold you can see is worth more to a planner than an ensemble you
  cannot.
- **ACV** ranks cars by how far each sits above its own cooling setpoint while cooling. Rank
  aggregation over five features was tested and scored worse; the report says why.

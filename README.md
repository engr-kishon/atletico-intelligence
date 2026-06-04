# Atletico Intelligence

AI-powered football video analysis that automatically detects players, classifies teams, tracks the ball, and delivers **offside/onside verdicts** for every pass — all from a raw match clip.

Upload a video, and the system returns an annotated video with team-colored overlays, a live mini pitch radar, and a per-pass decision table.

---

## About the Project

Atletico Intelligence solves a hard computer-vision problem: given a broadcast football video with no extra metadata, determine whether an attacker was in an offside position at the exact moment the ball was played.

The pipeline does this entirely from vision:

1. **Team classification** — samples frames across the clip, crops every detected player, and fits a visual clustering model to separate the two teams by kit color.
2. **Pitch homography** — detects 32 pitch keypoints per frame (penalty spots, corners, centre circle, etc.) and computes a perspective transform that maps any pixel on the grass to real-world centimetre coordinates on a standard pitch.
3. **Player tracking** — runs a YOLO detection model (ball, goalkeeper, player, referee) and ByteTrack multi-object tracker every frame. Team labels are assigned by majority vote over a rolling window to avoid flicker.
4. **Ball interpolation** — fills short gaps in ball detection and removes position outliers, then computes per-frame ball speed to distinguish genuine kicks from slow dribbles.
5. **Pass detection** — a possession change between two players of the same team, where the ball travels fast enough and long enough in flight, is recorded as a pass.
6. **Offside judgement** — at the kick frame, the two deepest defenders (plus the goalkeeper) define the defensive line in pitch coordinates. Any attacker beyond that line at the moment of the pass is flagged offside.
7. **Rendering** — writes an annotated MP4 with ellipses coloured by team, tracker IDs, a gold triangle on the ball, the offside/onside line drawn back onto the video frame, and a 30 % wide pitch radar overlay in the bottom-right corner.

---

## Architecture

```
Browser
  │
  └─► nginx :80
        ├── GET /            → React SPA (static files in /usr/share/nginx/html)
        └── /analyze-video
            /get-analysis-result      → FastAPI (uvicorn :8000)
            /uploads/*                    │
                                          ├── SQLite  (data/atletico.db)
                                          ├── Uploads (uploads/)
                                          └── Celery worker
                                                └── Redis :6379 (broker + result backend)
```

| Layer | Technology |
|---|---|
| Frontend | React 19, Vite 8, Tailwind CSS 4 |
| API | FastAPI, Uvicorn, Python 3.13 |
| Task queue | Celery 5, Redis |
| Database | SQLite via SQLAlchemy 2 |
| CV / ML | YOLO26m (Ultralytics), Supervision, OpenCV, PyTorch |
| Team classifier | Roboflow `sports` library — `TeamClassifier` |
| Pitch geometry | `SoccerPitchConfiguration` + OpenCV `findHomography` (RANSAC) |
| Packaging | uv, Docker (supervisord single-container) |

### Request lifecycle

```
POST /analyze-video
  → save video to disk
  → create AnalysisJob (status=pending) in SQLite
  → enqueue Celery task

Celery worker
  → fit TeamClassifier on sampled frames
  → collect per-frame detections + pitch projections
  → detect passes and judge offside
  → render annotated video
  → update AnalysisJob (status=completed, verdict=JSON)

GET /get-analysis-result?task_id=…
  → poll job status
  → on completed: return stats JSON + annotated video URL
```

### ML models

| Model | File | Purpose |
|---|---|---|
| Player detector | `models/best.pt` | Detects ball (0), goalkeeper (1), player (2), referee (3) |
| Pitch keypoint detector | `models/point_best.pt` | Detects 32 pitch landmarks for homography |

Both models are YOLO26m. Inference runs on CUDA → MPS → CPU, in that priority order.

---

## Datasets

### Player detection model
**Roboflow — Football Players Detection**
[`roboflow-jvuqo/football-players-detection-3zvbc` dataset 20](https://universe.roboflow.com/roboflow-jvuqo/football-players-detection-3zvbc/dataset/20)

Annotated broadcast frames with bounding boxes for ball, goalkeepers, outfield players, and referees. Used to train `best.pt`.

### Pitch keypoint model
**Roboflow — Football Field Detection**
[`roboflow-jvuqo/football-field-detection-f07vi` dataset 16](https://universe.roboflow.com/roboflow-jvuqo/football-field-detection-f07vi/dataset/16)

Annotated pitch images with 32 keypoint landmarks (corners, penalty spots, centre circle, etc.). Used to train `point_best.pt` for homography estimation.

### Match footage
**Kaggle — DFL Bundesliga 460 MP4 Videos**
[`saberghaderi/-dfl-bundesliga-460-mp4-videos-in-30sec-csv`](https://www.kaggle.com/datasets/saberghaderi/-dfl-bundesliga-460-mp4-videos-in-30sec-csv)

460 broadcast clips of 30 seconds each from Bundesliga matches, used for fine-tuning, evaluation, and testing the full analysis pipeline end-to-end.

---

## How to Run with Docker

### Prerequisites

- [Docker](https://docs.docker.com/get-docker/) and [Docker Compose](https://docs.docker.com/compose/) installed
- `backend/.env` populated with your API keys (see below)

### 1. Configure environment variables

`backend/.env` must contain:

```env
ROBOFLOW_API_KEY=your_key_here
HF_TOKEN=your_huggingface_token_here
PLAYER_MODEL_PATH=models/best.pt
POINT_MODEL_PATH=models/point_best.pt
```

> The Redis URL is injected automatically by docker-compose and does not need to be set manually.

### 2. Build and start

```bash
docker compose up --build
```

The first build takes several minutes — PyTorch and the ML dependencies are large.

### 3. Open the app

Navigate to [http://localhost](http://localhost) in your browser.

Upload a football match clip (MP4). The status will change from *Uploading* → *Analyzing* → *Completed*. The annotated video and decision table will appear automatically once processing finishes.

### 4. Stop

```bash
docker compose down
```

Uploaded videos and the SQLite database are persisted in `backend/uploads/` and `backend/data/` on your host machine, so they survive container restarts.

### Ports

| Port | Service |
|---|---|
| 80 | App (nginx — frontend + API proxy) |

All other ports (Redis 6379, uvicorn 8000) are internal to the container and not exposed to the host.

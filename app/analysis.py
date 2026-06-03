import os
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import logging
from collections import defaultdict, deque, Counter

import cv2
import numpy as np
import supervision as sv

from ultralytics import YOLO
from sports.common.team import TeamClassifier
from sports.configs.soccer import SoccerPitchConfiguration
from sports.annotators.soccer import draw_pitch, draw_points_on_pitch

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
PLAYER_MODEL_ID  = "football-players-detection-3zvbc/11"
PLAYER_MODEL_PATH = os.getenv("PLAYER_MODEL_PATH", "models/best.pt")
POINT_MODEL_PATH  = os.getenv("POINT_MODEL_PATH", "models/point_best.pt")

BALL_ID, GOALKEEPER_ID, PLAYER_ID, REFEREE_ID = 0, 1, 2, 3
CONF_THRESHOLD, NMS_THRESHOLD, STRIDE = 0.3, 0.5, 30

REVOTE_INTERVAL, VOTE_HISTORY, EMA_ALPHA = 15, 15, 0.4
PITCH_CONF, RANSAC_REPROJ = 0.5, 500.0

POSSESSION_DIST, KICK_SPEED = float(os.getenv("POSSESSION_DIST", 350)), float(os.getenv("KICK_SPEED", 50))
MIN_FLIGHT, MAX_FLIGHT, MAX_BALL_GAP = 3, 90, 10
LEVEL_TOL_CM, GK_MARGIN_FRAC = 40.0, 0.15

RADAR_SCALE_W, VERDICT_HOLD = 0.30, 30
GREEN, RED = (0, 200, 0), (0, 0, 255)

CONFIG = SoccerPitchConfiguration()
PITCH_VERTICES = np.array(CONFIG.vertices, dtype=np.float32)
PITCH_LENGTH = float(PITCH_VERTICES[:, 0].max())
PITCH_WIDTH  = float(PITCH_VERTICES[:, 1].max())
CENTER = PITCH_LENGTH / 2.0

_PALETTE  = sv.ColorPalette.from_hex(["#00BFFF", "#FF1493", "#FFD700"])
_ELLIPSE  = sv.EllipseAnnotator(color=_PALETTE, thickness=2)
_LABEL    = sv.LabelAnnotator(color=_PALETTE, text_color=sv.Color.from_hex("#000000"),
                              text_position=sv.Position.BOTTOM_CENTER)
_TRIANGLE = sv.TriangleAnnotator(color=sv.Color.from_hex("#FFD700"), base=25, height=21, outline_thickness=1)

try:
    import torch
    if torch.cuda.is_available():
        DEVICE = "cuda"
    elif torch.backends.mps.is_available():
        DEVICE = "mps"
    else:
        DEVICE = "cpu"
except Exception:
    DEVICE = "cpu"

HALF = DEVICE == "cuda"
PREDICT_KWARGS = dict(verbose=False, device=DEVICE, half=HALF)

TEAM_DEVICE = DEVICE

# --------------------------------------------------------------------------- #
# Lazy model singletons (player + pitch are reusable across tasks)
# --------------------------------------------------------------------------- #
_player_model = None
_pitch_model = None

def _get_player_model():
    global _player_model
    if _player_model is None:
        _player_model = YOLO(PLAYER_MODEL_PATH); _player_model.to(DEVICE)
        logger.info("Player model loaded: %s (%s)", PLAYER_MODEL_PATH, DEVICE)
    return _player_model

def _get_pitch_model():
    global _pitch_model
    if _pitch_model is None:
        _pitch_model = YOLO(POINT_MODEL_PATH); _pitch_model.to(DEVICE)
        logger.info("Pitch model loaded: %s (%s)", POINT_MODEL_PATH, DEVICE)
    return _pitch_model

# --------------------------------------------------------------------------- #
# Geometry / offside helpers
# --------------------------------------------------------------------------- #
def _anchors(d):
    return np.empty((0, 2)) if len(d) == 0 else d.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)

def _project(xy, H):
    if H is None or len(xy) == 0:
        return np.empty((0, 2))
    return cv2.perspectiveTransform(np.asarray(xy, np.float32).reshape(-1, 1, 2), H).reshape(-1, 2)

def _project_inv(pts, H_inv):
    return cv2.perspectiveTransform(np.asarray(pts, np.float32).reshape(-1, 1, 2), H_inv).reshape(-1, 2)

def _homography(pitch_result):
    kp = sv.KeyPoints.from_ultralytics(pitch_result)
    if len(kp.xy) == 0 or kp.confidence is None:
        return None
    mask = kp.confidence[0] > PITCH_CONF
    if int(mask.sum()) < 4:
        return None
    H, _ = cv2.findHomography(kp.xy[0][mask].astype(np.float32), PITCH_VERTICES[mask], cv2.RANSAC, RANSAC_REPROJ)
    return None if H is None else H / H[2, 2]

def _resolve_gk_team(players, gks):
    if len(gks) == 0:
        return np.array([])
    if len(players) == 0:
        return np.zeros(len(gks))
    p = players.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
    g = gks.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
    t0, t1 = p[players.class_id == 0], p[players.class_id == 1]
    if len(t0) == 0 or len(t1) == 0:
        return np.zeros(len(gks))
    c0, c1 = t0.mean(0), t1.mean(0)
    return np.array([0 if np.linalg.norm(x - c0) < np.linalg.norm(x - c1) else 1 for x in g])

def _majority(v):
    return Counter(v).most_common(1)[0][0]

def _draw_offside_line(frame, line_x, H, color, thickness=3):
    try:
        H_inv = np.linalg.inv(H)
    except np.linalg.LinAlgError:
        return
    ys = np.linspace(0, PITCH_WIDTH, 50)
    img = _project_inv(np.column_stack([np.full_like(ys, line_x), ys]), H_inv)
    h, w = frame.shape[:2]
    ok = (img[:, 0] > -5*w) & (img[:, 0] < 5*w) & (img[:, 1] > -5*h) & (img[:, 1] < 5*h)
    pts = img[ok].astype(np.int32)
    if len(pts) >= 2:
        cv2.polylines(frame, [pts], False, color, thickness, cv2.LINE_AA)

def _radar_line_points(line_x):
    ys = np.linspace(0, PITCH_WIDTH, 60)
    return np.column_stack([np.full_like(ys, line_x), ys])

# --------------------------------------------------------------------------- #
# Stages
# --------------------------------------------------------------------------- #
def _fit_team_classifier(video_path):
    model = _get_player_model()
    crops = []
    for frame in sv.get_video_frames_generator(source_path=video_path, stride=STRIDE):
        det = sv.Detections.from_ultralytics(model(frame, conf=CONF_THRESHOLD, **PREDICT_KWARGS)[0])
        players = det[det.class_id == PLAYER_ID]
        crops += [sv.crop_image(frame, b) for b in players.xyxy]
    tc = TeamClassifier(device=TEAM_DEVICE)
    tc.fit(crops)
    logger.info("TeamClassifier fitted on %d crops", len(crops))
    return tc

def _collect(video_path, team_classifier):
    model, pitch_model = _get_player_model(), _get_pitch_model()
    tracker = sv.ByteTrack(); tracker.reset()
    votes = defaultdict(lambda: deque(maxlen=VOTE_HISTORY))
    H_smooth, records = None, []

    for fidx, frame in enumerate(sv.get_video_frames_generator(video_path)):
        H_raw = _homography(pitch_model(frame, **PREDICT_KWARGS)[0])
        if H_raw is not None:
            H_smooth = H_raw if H_smooth is None else (EMA_ALPHA*H_raw + (1-EMA_ALPHA)*H_smooth)
            H_smooth = H_smooth / H_smooth[2, 2]

        det = sv.Detections.from_ultralytics(model(frame, conf=CONF_THRESHOLD, **PREDICT_KWARGS)[0])
        ball = det[det.class_id == BALL_ID]
        if len(ball):
            ball.xyxy = sv.pad_boxes(ball.xyxy, px=10)
        det = det[det.class_id != BALL_ID].with_nms(threshold=NMS_THRESHOLD, class_agnostic=True)
        det = tracker.update_with_detections(det)

        gk      = det[det.class_id == GOALKEEPER_ID]
        players = det[det.class_id == PLAYER_ID]
        refs    = det[det.class_id == REFEREE_ID]

        if len(players):
            revote = (fidx % REVOTE_INTERVAL == 0)
            need = [i for i, t in enumerate(players.tracker_id) if t not in votes or revote]
            if need:
                preds = team_classifier.predict([sv.crop_image(frame, players.xyxy[i]) for i in need])
                for i, p in zip(need, preds):
                    votes[players.tracker_id[i]].append(int(p))
            players.class_id = np.array([_majority(votes[t]) for t in players.tracker_id])
        if len(gk):
            gk.class_id = _resolve_gk_team(players, gk)
        if len(refs):
            refs.class_id = np.full(len(refs), 2)

        bxy = _project(_anchors(ball), H_smooth)
        records.append({
            "players_box":  players.xyxy.copy() if len(players) else np.empty((0, 4)),
            "players_team": np.asarray(players.class_id) if len(players) else np.empty(0, int),
            "players_tid":  np.asarray(players.tracker_id) if len(players) else np.empty(0, int),
            "players_pitch": _project(_anchors(players), H_smooth),
            "gk_box":  gk.xyxy.copy() if len(gk) else np.empty((0, 4)),
            "gk_team": np.asarray(gk.class_id) if len(gk) else np.empty(0, int),
            "gk_tid":  np.asarray(gk.tracker_id) if len(gk) else np.empty(0, int),
            "gk_pitch": _project(_anchors(gk), H_smooth),
            "ref_box": refs.xyxy.copy() if len(refs) else np.empty((0, 4)),
            "ref_tid": np.asarray(refs.tracker_id) if len(refs) else np.empty(0, int),
            "ball_box":  (ball.xyxy[0].copy() if len(ball) else None),
            "ball_pitch": (bxy[0] if len(bxy) else None),
            "H": H_smooth,
        })
    logger.info("Collected %d frames", len(records))
    return records

def _estimate_attack_sign(records):
    gk_x = {0: [], 1: []}; plr_x = {0: [], 1: []}
    for r in records:
        if r["H"] is None:
            continue
        for t, xy in zip(r["players_team"], r["players_pitch"]):
            plr_x[int(t)].append(float(xy[0]))
        for t, xy in zip(r["gk_team"], r["gk_pitch"]):
            gk_x[int(t)].append(float(xy[0]))

    def m(v): return float(np.mean(v)) if len(v) else None
    gk0, gk1, p0, p1 = m(gk_x[0]), m(gk_x[1]), m(plr_x[0]), m(plr_x[1])
    margin = GK_MARGIN_FRAC * PITCH_LENGTH

    if gk0 is not None and gk1 is not None and abs(gk0 - gk1) > 0.10 * PITCH_LENGTH:
        team0_right = gk0 < gk1
    elif gk0 is not None and abs(gk0 - CENTER) > margin:
        team0_right = gk0 < CENTER
    elif gk1 is not None and abs(gk1 - CENTER) > margin:
        team0_right = not (gk1 < CENTER)
    elif p0 is not None and p1 is not None:
        team0_right = p0 < p1
    else:
        team0_right = True
    sign = {0: +1, 1: -1} if team0_right else {0: -1, 1: +1}
    logger.info("Attack sign: %s", sign)
    return sign

def _offside_ids_at(rec, attacking_team, ball_xy, sign):
    s, dfd = sign[attacking_team], 1 - attacking_team
    def_adv = list(s * rec["players_pitch"][rec["players_team"] == dfd][:, 0])
    def_adv += list(s * rec["gk_pitch"][rec["gk_team"] == dfd][:, 0])
    if len(def_adv) < 2:
        return set(), None
    line_adv = max(np.sort(def_adv)[-2], s * ball_xy[0])
    half_adv = s * CENTER
    off = set()
    m = rec["players_team"] == attacking_team
    for tid, xy in zip(rec["players_tid"][m], rec["players_pitch"][m]):
        a = s * xy[0]
        if a > half_adv and (a - line_adv) > LEVEL_TOL_CM:
            off.add(int(tid))
    return off, s * line_adv

def _detect_and_judge(records, sign):
    N = len(records)
    ball = np.full((N, 2), np.nan)
    for i, r in enumerate(records):
        if r["ball_pitch"] is not None:
            ball[i] = r["ball_pitch"]
    valid = np.where(~np.isnan(ball[:, 0]))[0]
    for a, b in zip(valid[:-1], valid[1:]):
        if 1 < b - a <= MAX_BALL_GAP + 1:
            for k in range(a + 1, b):
                t = (k - a) / (b - a); ball[k] = (1 - t) * ball[a] + t * ball[b]
    for i in range(1, N - 1):
        if np.isnan(ball[i, 0]) or np.isnan(ball[i-1, 0]) or np.isnan(ball[i+1, 0]):
            continue
        if np.hypot(*(ball[i]-ball[i-1])) > 400 and np.hypot(*(ball[i]-ball[i+1])) > 400:
            ball[i] = 0.5 * (ball[i-1] + ball[i+1])

    speed = np.full(N, np.nan)
    for i in range(1, N):
        if not np.isnan(ball[i, 0]) and not np.isnan(ball[i-1, 0]):
            speed[i] = np.hypot(*(ball[i] - ball[i-1]))

    def possessor(i):
        if np.isnan(ball[i, 0]) or len(records[i]["players_pitch"]) == 0:
            return None
        d = np.linalg.norm(records[i]["players_pitch"] - ball[i], axis=1)
        j = int(np.argmin(d))
        if d[j] < POSSESSION_DIST:
            return int(records[i]["players_tid"][j]), int(records[i]["players_team"][j])
        return None

    segs, cur = [], None
    for i in range(N):
        p = possessor(i)
        if p is None:
            continue
        if cur is None or p[0] != cur[2]:
            if cur is not None:
                segs.append(cur)
            cur = [i, i, p[0], p[1]]
        else:
            cur[1] = i
    if cur is not None:
        segs.append(cur)

    verdicts, n_passes = [], 0
    for a, b in zip(segs[:-1], segs[1:]):
        kick_f, recv_f = a[1], b[0]
        gap = recv_f - kick_f
        if gap < MIN_FLIGHT or gap > MAX_FLIGHT:
            continue
        seg = speed[kick_f + 1:recv_f + 1]
        if not (np.any(~np.isnan(seg)) and np.nanmax(seg) > KICK_SPEED):
            continue
        if np.isnan(ball[kick_f, 0]):
            continue
        n_passes += 1
        if b[3] != a[3]:
            continue
        off_ids, line_x = _offside_ids_at(records[kick_f], a[3], ball[kick_f], sign)
        verdicts.append({
            "kick_f": int(kick_f), "passer": int(a[2]), "team": int(a[3]),
            "recv_f": int(recv_f), "recv": int(b[2]),
            "line_x": (None if line_x is None else float(line_x)),
            "offside_ids": off_ids, "offence": int(b[2]) in off_ids,
        })
    return verdicts, n_passes

# --------------------------------------------------------------------------- #
# Render
# --------------------------------------------------------------------------- #
def _to_dets(box, cls, tid):
    if len(box) == 0:
        return sv.Detections.empty()
    return sv.Detections(xyxy=box.astype(np.float32), class_id=cls.astype(int), tracker_id=tid.astype(int))

def _build_radar(rec, line_x=None):
    radar = draw_pitch(CONFIG)
    for team, hexc in ((0, '00BFFF'), (1, 'FF1493')):
        xy = rec["players_pitch"][rec["players_team"] == team] if len(rec["players_pitch"]) else np.empty((0, 2))
        radar = draw_points_on_pitch(config=CONFIG, xy=xy, face_color=sv.Color.from_hex(hexc),
                                     edge_color=sv.Color.BLACK, radius=16, pitch=radar)
    if len(rec["gk_pitch"]):
        radar = draw_points_on_pitch(config=CONFIG, xy=rec["gk_pitch"], face_color=sv.Color.from_hex('FFD700'),
                                     edge_color=sv.Color.BLACK, radius=16, pitch=radar)
    if rec["ball_pitch"] is not None:
        radar = draw_points_on_pitch(config=CONFIG, xy=np.array([rec["ball_pitch"]]), face_color=sv.Color.WHITE,
                                     edge_color=sv.Color.BLACK, radius=10, pitch=radar)
    if line_x is not None:
        radar = draw_points_on_pitch(config=CONFIG, xy=_radar_line_points(line_x), face_color=sv.Color.from_hex('FF3030'),
                                     edge_color=sv.Color.from_hex('FF3030'), radius=4, pitch=radar)
    return radar

def _overlay_radar(frame, radar, margin=20, border=3):
    fh, fw = frame.shape[:2]
    rw = int(fw * RADAR_SCALE_W); rh = int(radar.shape[0] * rw / radar.shape[1])
    small = cv2.resize(radar, (rw, rh))
    x2, y2 = fw - margin, fh - margin; x1, y1 = x2 - rw, y2 - rh
    cv2.rectangle(frame, (x1-border, y1-border), (x2+border, y2+border), (255, 255, 255), border)
    frame[y1:y2, x1:x2] = small
    return frame

def _banner(frame, text, color):
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 3)
    cv2.rectangle(frame, (20, 18), (44 + tw, 40 + th), (0, 0, 0), -1)
    cv2.putText(frame, text, (32, 32 + th), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 3, cv2.LINE_AA)

def _render(video_path, output_path, records, verdicts):
    active = {}
    for v in verdicts:
        for f in range(v["kick_f"], v["kick_f"] + VERDICT_HOLD + 1):
            active[f] = v

    video_info = sv.VideoInfo.from_video_path(video_path)
    with sv.VideoSink(output_path, video_info) as sink:
        for fidx, frame in enumerate(sv.get_video_frames_generator(video_path)):
            rec = records[fidx]
            players = _to_dets(rec["players_box"], rec["players_team"], rec["players_tid"])
            gk      = _to_dets(rec["gk_box"], rec["gk_team"], rec["gk_tid"])
            refs    = _to_dets(rec["ref_box"], np.full(len(rec["ref_box"]), 2), rec["ref_tid"])
            merged  = sv.Detections.merge([players, gk, refs])

            annotated = frame.copy()
            if len(merged):
                merged.class_id = merged.class_id.astype(int)
                annotated = _ELLIPSE.annotate(annotated, merged)
                annotated = _LABEL.annotate(annotated, merged, [f"#{t}" for t in merged.tracker_id])
            if rec["ball_box"] is not None:
                annotated = _TRIANGLE.annotate(annotated, sv.Detections(
                    xyxy=rec["ball_box"][None, :].astype(np.float32), class_id=np.array([0])))

            v = active.get(fidx)
            line_x = v["line_x"] if (v and v["line_x"] is not None) else None
            if v and line_x is not None and rec["H"] is not None:
                color = RED if v["offence"] else GREEN
                _draw_offside_line(annotated, line_x, rec["H"], color, 3)
                hit = np.where(rec["players_tid"] == v["recv"])[0]
                if len(hit):
                    x1, y1, x2, y2 = rec["players_box"][hit[0]].astype(int)
                    cv2.ellipse(annotated, ((x1+x2)//2, y2), ((x2-x1)//2, 16), 0, 0, 360, color, 5)
                _banner(annotated, f"{'OFFSIDE' if v['offence'] else 'ONSIDE'}  #{v['passer']} -> #{v['recv']}", color)

            annotated = _overlay_radar(annotated, _build_radar(rec, line_x))
            sink.write_frame(annotated)
    logger.info("Rendered annotated video -> %s", output_path)

# --------------------------------------------------------------------------- #
# Public entry point (called by the Celery worker)
# --------------------------------------------------------------------------- #
def run_analysis(video_path: str, output_path: str) -> dict:
    logger.info("run_analysis: %s -> %s", video_path, output_path)

    team_classifier = _fit_team_classifier(video_path)      # 1 + 2: input + analysis
    records = _collect(video_path, team_classifier)          # 3: detect / track / project
    sign = _estimate_attack_sign(records)                    # attack direction
    verdicts, n_passes = _detect_and_judge(records, sign)    # 5: offside / onside
    _render(video_path, output_path, records, verdicts)      # 4 + 7: radar + saved video

    n_off = sum(v["offence"] for v in verdicts)
    stats = {
        "players_detected": max((len(r["players_box"]) for r in records), default=0),
        "ball_detected": any(r["ball_box"] is not None for r in records),
        "passes_detected": n_passes,
        "teammate_passes": len(verdicts),
        "offside_offences": n_off,
        "decisions": [{
            "kick_frame": v["kick_f"], "passer": v["passer"], "receiver": v["recv"],
            "attackers_beyond_line": len(v["offside_ids"]),
            "verdict": "offside" if v["offence"] else "onside",
        } for v in verdicts],
    }
    logger.info("run_analysis done: %d passes, %d offences", n_passes, n_off)
    return stats
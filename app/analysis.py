"""
Football analysis pipeline — Roboflow Inference + Supervision.

Model: football-players-detection-3zvbc/11 (4-class detection)
  0: ball | 1: goalkeeper | 2: player | 3: referee

Requires ROBOFLOW_API_KEY environment variable.
"""

import logging
import os

import numpy as np
import supervision as sv
from inference import get_model
from sports.common.team import TeamClassifier

logger = logging.getLogger(__name__)

PLAYER_DETECTION_MODEL_ID = "football-players-detection-3zvbc/11"
BALL_ID = 0
GOALKEEPER_ID = 1
PLAYER_ID = 2
REFEREE_ID = 3

CONF_THRESHOLD = 0.3
NMS_THRESHOLD = 0.5
STRIDE = 30  # sample every Nth frame when collecting team-classifier training data

# Video game style palette: team0=cyan, team1=pink, referee=gold
_PALETTE = sv.ColorPalette.from_hex(["#00BFFF", "#FF1493", "#FFD700"])


def _resolve_goalkeeper_teams(players: sv.Detections, goalkeepers: sv.Detections) -> np.ndarray:
    if len(players) == 0:
        return np.zeros(len(goalkeepers), dtype=int)
    players_xy = players.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
    team_ids = []
    for gk_xy in goalkeepers.get_anchors_coordinates(sv.Position.BOTTOM_CENTER):
        t0 = players_xy[players.class_id == 0]
        t1 = players_xy[players.class_id == 1]
        if len(t0) and len(t1):
            team_ids.append(0 if np.linalg.norm(gk_xy - t0.mean(0)) < np.linalg.norm(gk_xy - t1.mean(0)) else 1)
        else:
            team_ids.append(0 if len(t0) else 1)
    return np.array(team_ids, dtype=int)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_analysis(video_path: str, output_path: str) -> dict:
    model = get_model(
        model_id=PLAYER_DETECTION_MODEL_ID,
        api_key=os.getenv("ROBOFLOW_API_KEY"),
    )
    logger.info("Roboflow model loaded: %s", PLAYER_DETECTION_MODEL_ID)

    video_info = sv.VideoInfo.from_video_path(video_path)
    logger.info(
        "Video: %s  %dx%d  %.2f fps  %d frames",
        video_path, video_info.width, video_info.height,
        video_info.fps, video_info.total_frames,
    )

    # Phase 1: collect player crops across strided frames → fit TeamClassifier
    logger.info("Phase 1: collecting player crops (stride=%d)", STRIDE)
    crops = []
    for frame in sv.get_video_frames_generator(source_path=video_path, stride=STRIDE):
        result = model.infer(frame, confidence=CONF_THRESHOLD)[0]
        detections = sv.Detections.from_inference(result)
        players_crops = [
            sv.crop_image(frame, xyxy)
            for xyxy in detections[detections.class_id == PLAYER_ID].xyxy
        ]
        crops += players_crops

    logger.info("Phase 1 complete: %d player crops collected", len(crops))
    team_classifier = TeamClassifier(device="cpu")
    team_classifier.fit(crops)
    logger.info("TeamClassifier fitted (SigLIP + UMAP + KMeans)")

    # Annotators — video game style
    ellipse_ann = sv.EllipseAnnotator(color=_PALETTE, thickness=2)
    label_ann = sv.LabelAnnotator(
        color=_PALETTE,
        text_color=sv.Color.from_hex("#000000"),
        text_position=sv.Position.BOTTOM_CENTER,
    )
    triangle_ann = sv.TriangleAnnotator(
        color=sv.Color.from_hex("#FFD700"),
        base=25, height=21, outline_thickness=1,
    )

    tracker = sv.ByteTrack()
    tracker.reset()

    stats: dict = {"players_detected": 0, "ball_detected": False}

    # Phase 2: process every frame → annotate → write
    logger.info("Phase 2: processing all frames → %s", output_path)
    total_frames = video_info.total_frames
    _progress_logged: set[int] = set()
    with sv.VideoSink(output_path, video_info) as sink:
        for frame_idx, frame in enumerate(sv.get_video_frames_generator(video_path)):
            result = model.infer(frame, confidence=CONF_THRESHOLD)[0]
            det = sv.Detections.from_inference(result)

            ball_det = det[det.class_id == BALL_ID]
            ball_det.xyxy = sv.pad_boxes(xyxy=ball_det.xyxy, px=10)

            people_det = det[det.class_id != BALL_ID]
            people_det = people_det.with_nms(threshold=NMS_THRESHOLD, class_agnostic=True)
            people_det = tracker.update_with_detections(people_det)

            gk_det = people_det[people_det.class_id == GOALKEEPER_ID]
            player_det = people_det[people_det.class_id == PLAYER_ID]
            ref_det = people_det[people_det.class_id == REFEREE_ID]

            if len(player_det) > 0:
                players_crops = [sv.crop_image(frame, xyxy) for xyxy in player_det.xyxy]
                player_det.class_id = team_classifier.predict(players_crops)

            if len(gk_det) > 0:
                gk_det.class_id = _resolve_goalkeeper_teams(player_det, gk_det)

            if len(ref_det) > 0:
                ref_det.class_id = ref_det.class_id - 1  # 3 → 2 (gold palette index)

            non_empty = [d for d in [player_det, gk_det, ref_det] if len(d) > 0]
            all_det = sv.Detections.merge(non_empty) if non_empty else sv.Detections.empty()
            all_det.class_id = all_det.class_id.astype(int)

            tracker_ids = all_det.tracker_id
            labels = [f"#{tid}" for tid in tracker_ids] if tracker_ids is not None else []

            annotated = ellipse_ann.annotate(frame.copy(), all_det)
            annotated = label_ann.annotate(annotated, all_det, labels)
            annotated = triangle_ann.annotate(annotated, ball_det)
            sink.write_frame(annotated)

            stats["players_detected"] = max(stats["players_detected"], len(player_det))
            if len(ball_det) > 0:
                stats["ball_detected"] = True

            if total_frames > 0:
                pct = int((frame_idx + 1) / total_frames * 100)
                milestone = (pct // 10) * 10
                if milestone > 0 and milestone not in _progress_logged:
                    _progress_logged.add(milestone)
                    logger.info("Phase 2 progress: %d%% (%d/%d frames)", milestone, frame_idx + 1, total_frames)

    logger.info(
        "Analysis complete: players_detected=%d ball_detected=%s output=%s",
        stats["players_detected"], stats["ball_detected"], output_path,
    )
    return stats

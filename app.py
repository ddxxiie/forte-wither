from __future__ import annotations

import base64
import csv
import json
import os
import secrets
import threading
import time
import uuid
from pathlib import Path
from typing import Any

import cv2
import joblib
import numpy as np
from fastapi import Depends, FastAPI, File, Header, HTTPException, UploadFile
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.metrics import balanced_accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import GroupShuffleSplit

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
UPLOADS = DATA / "uploads"
EVIDENCE = DATA / "evidence"
MODEL_PATH = DATA / "model.joblib"
API_KEY_PATH = ROOT / ".api_key"
for directory in (UPLOADS, EVIDENCE):
    directory.mkdir(parents=True, exist_ok=True)
MAX_VIDEO_BYTES = int(os.getenv("MAX_VIDEO_MB", "300")) * 1024 * 1024
SAMPLE_FPS = 4
PATCH_SIZE = (64, 64)
WINDOW_SECONDS = 2.0
RULE = {
    "title": "Правила кассовых операций и инкассации — постановление Правления НБРК № 231 от 29.11.2019",
    "document": "Учитывает изменения, вступившие в силу 20.07.2026; актуальность проверена 18.10.2026",
    "url": "https://adilet.zan.kz/rus/docs/V1900019680",
    "status": "Действует на дату проверки 18.10.2026",
}
VIOLATION_TYPES = {
    1: "Передача ценностей вне специального устройства",
    2: "Нарушение порядка пересчёта наличных",
    3: "Оставление денежных средств без присмотра",
    4: "Открытый или незапертый сейф",
    5: "Оставление рабочего места без блокировки",
    6: "Несанкционированный доступ посторонних лиц в кассовую зону",
    7: "Оставление инкассаторской сумки без присмотра",
    8: "Смешивание денежных средств разных операций",
    9: "Посторонние предметы, еда и напитки на рабочем месте",
    10: "Оставление документов и печатей без присмотра",
    11: "Передача ценностей неуполномоченному лицу",
}

# Legal mapping is deliberately conservative: video can flag a visual cue, but
# cannot establish authorization, intent, bank policy, or the complete facts.
LEGAL_BASES = {
    1: {"status": "normative_candidate", "label": "Есть требования к хранению и передаче наличных; факт требует проверки", "document": RULE["title"], "paragraphs": "пп. 20, 23 (спецустройства должны закрываться на ключ; передача принятых денег заведующему кассой оформляется по кассовым документам)", "url": RULE["url"], "note": "В кадре можно заметить способ передачи, но нельзя подтвердить вид ценностей, этап операции и ее документальное оформление."},
    2: {"status": "normative_candidate", "label": "Есть применимые требования НПА; факт требует проверки", "document": RULE["title"], "paragraphs": "пп. 3(4), 41–44", "url": RULE["url"], "note": "Правила устанавливают пересчет банкнот по листам, монет по кружкам, использование оборудования и присутствие контролирующего работника в предусмотренных случаях. Видео само по себе не доказывает ошибку пересчета."},
    3: {"status": "normative_candidate", "label": "Есть требование к сохранности; факт требует проверки", "document": RULE["title"], "paragraphs": "пп. 3(7), 74, 78–80", "url": RULE["url"], "note": "НПА требует обеспечить сохранность и хранение наличных в кладовой и (или) сейфовой комнате с учетом внутренних документов. Отсутствие человека рядом в кадре само по себе не подтверждает оставление без присмотра."},
    4: {"status": "normative_candidate", "label": "Есть прямое требование к хранению наличных; факт требует проверки", "document": RULE["title"], "paragraphs": "пп. 20, 74, 80", "url": RULE["url"], "note": "Ранее принятые деньги должны храниться в спецустройствах, закрывающихся на ключ; детали порядка обращения с замками и ключами закрепляются банком. Открытый сейф в кадре не доказывает, что в нем находятся деньги или что он оставлен без контроля."},
    5: {"status": "normative_candidate", "label": "Есть прямое требование по окончанию обслуживания; факт требует проверки", "document": RULE["title"], "paragraphs": "пп. 20, 22", "url": RULE["url"], "note": "Кассиру запрещено покидать рабочее место до окончания обслуживания клиента; принятые ранее деньги хранятся в устройствах, закрывающихся на ключ. Видео не всегда позволяет установить, завершено ли обслуживание и где находятся деньги."},
    6: {"status": "conditional_norm", "label": "Норма применяется при установленной сфере действия", "document": "Инструкция по антитеррористической защите расчетно-кассовых отделений банков, уязвимых в террористическом отношении", "paragraphs": "пп. 13–14", "url": "https://adilet.zan.kz/rus/docs/V2400034348", "note": "Правила пропускного режима относятся к уязвимым в террористическом отношении расчетно-кассовым отделениям. Для иных кассовых зон полномочия и допуск нужно сверять с применимыми внутренними документами и режимом доступа."},
    7: {"status": "normative_candidate", "label": "Есть требования к сохранности при инкассации; факт требует проверки", "document": RULE["title"], "paragraphs": "пп. 109–121 (операции и хранение при инкассации)", "url": RULE["url"], "note": "Правила регулируют прием, перевозку и хранение ценностей инкассаторами. Кадр с сумкой без человека рядом не устанавливает ответственное лицо, момент передачи или соблюдение маршрута."},
    8: {"status": "normative_candidate", "label": "Есть требование выполнять операции раздельно; факт требует проверки", "document": RULE["title"], "paragraphs": "пп. 20–21", "url": RULE["url"], "note": "Наличные клиента находятся на рабочем месте кассира, а при нескольких приходных документах прием ведется отдельно по каждому документу. Чтобы установить смешивание разных операций, одного видео может быть недостаточно — нужны документы учета."},
    9: {"status": "conditional_norm", "label": "Есть норма о личных вещах; отдельный запрет еды требует подтверждения", "document": RULE["title"], "paragraphs": "п. 12", "url": RULE["url"], "note": "Личные вещи работников хранятся вне кассового узла; вносить туда разрешено перечисленные исключения. В проверенных актах отдельный прямой запрет на еду и напитки не найден, поэтому категория отмечается как визуальный сигнал, а не установленное нарушение."},
    10: {"status": "internal_policy", "label": "Сверить с внутренним порядком банка", "document": RULE["title"], "paragraphs": "п. 80", "url": RULE["url"], "note": "Порядок хранения, выдачи и передачи печатей, штампов, ключей и кодов устанавливают внутренние документы банка. Видео может лишь указать на необходимость проверки доступа и фактического порядка хранения."},
    11: {"status": "normative_candidate", "label": "Есть требования к ролям и документированию; факт требует проверки", "document": RULE["title"], "paragraphs": "пп. 14–15", "url": RULE["url"], "note": "Правила закрепляют выполнение операций согласно должностным обязанностям и документирование передачи наличных между кассовыми работниками. Для вывода о неуполномоченном получателе нужны сведения о его роли и полномочиях."},
}


def load_api_key() -> str:
    value = os.getenv("API_KEY", "").strip()
    if value:
        return value
    if API_KEY_PATH.exists():
        return API_KEY_PATH.read_text(encoding="utf-8").strip()
    value = secrets.token_urlsafe(32)
    API_KEY_PATH.write_text(value, encoding="utf-8")
    try:
        API_KEY_PATH.chmod(0o600)
    except OSError:
        pass
    return value


API_KEY = load_api_key()
app = FastAPI(title="Forte Wither — локальная проверка видео", version="1.1.0")
jobs: dict[str, dict[str, Any]] = {}
jobs_lock = threading.Lock()
model_lock = threading.Lock()
hog = cv2.HOGDescriptor(_winSize=PATCH_SIZE, _blockSize=(16, 16), _blockStride=(8, 8), _cellSize=(8, 8), _nbins=9)
model_bundle: dict[str, Any] | None = None


def require_key(x_api_key: str | None = Header(default=None, alias="X-API-Key")) -> None:
    if not x_api_key or not secrets.compare_digest(x_api_key, API_KEY):
        raise HTTPException(status_code=401, detail="Неверный или отсутствующий X-API-Key")


def frame_features(frame: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, PATCH_SIZE, interpolation=cv2.INTER_AREA)
    # HOG keeps the baseline small and entirely CPU/local; normalized color is included for scene cues.
    descriptor = hog.compute(gray).reshape(-1).astype(np.float32)
    hist = cv2.calcHist([cv2.resize(frame, (32, 32))], [0, 1, 2], None, [4, 4, 4], [0, 256, 0, 256, 0, 256]).reshape(-1)
    hist = cv2.normalize(hist, None).reshape(-1).astype(np.float32)
    return np.concatenate((descriptor, hist))


def clip_feature(cap: cv2.VideoCapture, center: float, duration: float) -> np.ndarray | None:
    vectors: list[np.ndarray] = []
    gray_frames: list[np.ndarray] = []
    start = max(0.0, center - WINDOW_SECONDS / 2)
    end = min(duration, center + WINDOW_SECONDS / 2)
    if end <= start:
        return None
    for timestamp in np.linspace(start, end, SAMPLE_FPS, endpoint=False):
        cap.set(cv2.CAP_PROP_POS_MSEC, float(timestamp * 1000))
        ok, frame = cap.read()
        if ok:
            vectors.append(frame_features(frame))
            gray_frames.append(cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (64, 64), interpolation=cv2.INTER_AREA))
    if not vectors:
        return None
    arr = np.stack(vectors)
    # Mean and standard deviation encode appearance and change within the short temporal window.
    motion: list[float] = []
    for previous, current in zip(gray_frames, gray_frames[1:]):
        flow = cv2.calcOpticalFlowFarneback(previous, current, None, 0.5, 2, 15, 2, 5, 1.1, 0)
        magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1], angleInDegrees=False)
        for gy in range(4):
            for gx in range(4):
                ys, xs = slice(gy * 16, (gy + 1) * 16), slice(gx * 16, (gx + 1) * 16)
                hist, _ = np.histogram(angle[ys, xs], bins=8, range=(0, 2 * np.pi), weights=magnitude[ys, xs])
                motion.extend((hist / (hist.sum() + 1e-6)).tolist())
                motion.append(float(np.mean(magnitude[ys, xs])))
    return np.concatenate((arr.mean(axis=0), arr.std(axis=0), np.asarray(motion, dtype=np.float32)))


def video_info(path: Path) -> tuple[float, float]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError("Не удалось открыть видео. Проверьте, что это корректный .mov-файл.")
    fps = cap.get(cv2.CAP_PROP_FPS) or 0
    count = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
    duration = count / fps if fps else 0
    cap.release()
    if duration <= 0:
        raise ValueError("Не удалось определить длительность видео.")
    return duration, fps


def training_group_key(video_path: Path) -> str:
    """Keep an original recording and its mirrored copy in the same validation group."""
    stem = video_path.stem.translate(str.maketrans({"Х": "X", "х": "x"}))
    return stem.casefold()


def load_model() -> dict[str, Any] | None:
    global model_bundle
    with model_lock:
        if model_bundle is None and MODEL_PATH.exists():
            try:
                model_bundle = joblib.load(MODEL_PATH)
            except Exception:
                model_bundle = None
        return model_bundle


def jpg_base64(frame: np.ndarray) -> str:
    ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if not ok:
        return ""
    return base64.b64encode(encoded.tobytes()).decode("ascii")


def update_job_progress(job_id: str, progress: int, message: str) -> None:
    with jobs_lock:
        if job_id in jobs:
            jobs[job_id].update({"progress_percent": min(99, max(0, progress)), "progress_message": message})


def analyze_video(path: Path, job_id: str) -> dict[str, Any]:
    duration, fps = video_info(path)
    bundle = load_model()
    if bundle is None:
        raise ValueError("Модель ещё не обучена. Сначала положите размеченные видео в data/videos и выполните обучение по README.")
    cap = cv2.VideoCapture(str(path))
    stride = max(0.5, float(bundle.get("stride_sec", 0.75)))
    threshold = float(os.getenv("DETECTION_THRESHOLD", bundle.get("threshold", 0.35)))
    scored: list[tuple[float, float, np.ndarray]] = []
    total_windows = max(1, int(np.ceil((duration - WINDOW_SECONDS / 2) / stride)))
    processed_windows = 0
    t = WINDOW_SECONDS / 2
    while t < duration:
        feat = clip_feature(cap, t, duration)
        if feat is not None:
            prob = float(bundle["model"].predict_proba(feat.reshape(1, -1))[0, 1])
            scored.append((t, prob, feat))
        processed_windows += 1
        update_job_progress(job_id, 5 + int(90 * processed_windows / total_windows), "Анализ кадров и поиск событий")
        t += stride
    cap.release()
    # Collapse adjacent positive windows into events and retain the strongest evidence timestamp.
    positives = [(t, p, feat) for t, p, feat in scored if p >= threshold]
    groups: list[list[tuple[float, float, np.ndarray]]] = []
    for item in positives:
        if not groups or item[0] - groups[-1][-1][0] > max(2.0, stride * 2.1):
            groups.append([item])
        else:
            groups[-1].append(item)
    violations = []
    for group in groups:
        timestamp, confidence, event_features = max(group, key=lambda it: it[1])
        type_id = None
        type_name = "Возможное визуальное событие — тип не определён"
        type_confidence = None
        if bundle.get("type_model") is not None:
            type_probs = bundle["type_model"].predict_proba(event_features.reshape(1, -1))[0]
            best_type_index = int(np.argmax(type_probs))
            type_id = int(bundle["type_model"].classes_[best_type_index])
            type_name = VIOLATION_TYPES.get(type_id, "Тип требует проверки")
            type_confidence = round(float(type_probs[best_type_index]), 3)
        cap = cv2.VideoCapture(str(path))
        frames = []
        for offset in (-0.5, 0, 0.5):
            stamp = min(duration, max(0, timestamp + offset))
            cap.set(cv2.CAP_PROP_POS_MSEC, stamp * 1000)
            ok, frame = cap.read()
            if ok:
                frames.append({"timestamp_sec": round(stamp, 2), "image_base64": jpg_base64(frame)})
        cap.release()
        legal_basis = LEGAL_BASES.get(type_id, {"status": "unverified", "label": "Требуется правовая проверка", "document": "", "paragraphs": "", "url": RULE["url"], "note": "Не удалось сопоставить категорию с проверенной нормой."})
        violations.append({"type_id": type_id, "type": type_name, "type_confidence": type_confidence,
                           "timecode_sec": round(timestamp, 2),
                           "description": f"Модель выделила визуально похожий фрагмент (оценка модели {confidence:.0%}). Это не установленное нарушение. Категория {type_id if type_id is not None else 'не определена'}: {type_name}. {legal_basis['note']} Проверьте видео, полномочия участников, документы и применимые внутренние правила.",
                           "evidence_frames": frames[:3], "legal_qualification": legal_basis})
    return {"job_id": job_id, "status": "done", "verdict": "violation" if violations else "no_violation",
            "violations": violations, "video_duration_sec": round(duration, 2), "fps": round(fps, 2),
            "model_trained": True, "progress_percent": 100, "progress_message": "Обработка завершена",
            "notice": "Результат — визуальный сигнал для проверки, а не юридическая квалификация. Нормативная база проверена 18.10.2026 по ИПС Әділет; применимость зависит от вида подразделения, фактов и действующих внутренних документов банка. Категории 10 и 11 не представлены в размеченных обучающих видео."}


def run_job(job_id: str, path: Path) -> None:
    with jobs_lock:
        jobs[job_id] = {"job_id": job_id, "status": "processing", "progress_percent": 2, "progress_message": "Подготовка видео"}
    try:
        result = analyze_video(path, job_id)
        with jobs_lock:
            jobs[job_id] = result
    except Exception as exc:
        with jobs_lock:
            jobs[job_id] = {"job_id": job_id, "status": "failed", "error": str(exc)}
    finally:
        try:
            path.unlink(missing_ok=True)
        except OSError:
            pass


@app.post("/api/v1/analyze", dependencies=[Depends(require_key)])
async def create_analysis(file: UploadFile = File(...)) -> dict[str, str]:
    if Path(file.filename or "").suffix.lower() != ".mov":
        raise HTTPException(status_code=415, detail="Поддерживается только видео .mov")
    job_id = uuid.uuid4().hex[:12]
    path = UPLOADS / f"{job_id}.mov"
    size = 0
    try:
        with path.open("wb") as out:
            while chunk := await file.read(1024 * 1024):
                size += len(chunk)
                if size > MAX_VIDEO_BYTES:
                    raise HTTPException(status_code=413, detail=f"Максимальный размер видео — {MAX_VIDEO_BYTES // (1024 * 1024)} МБ")
                out.write(chunk)
    except Exception:
        path.unlink(missing_ok=True)
        raise
    finally:
        await file.close()
    with jobs_lock:
        jobs[job_id] = {"job_id": job_id, "status": "processing", "progress_percent": 0, "progress_message": "Видео загружено, анализ запущен"}
    threading.Thread(target=run_job, args=(job_id, path), daemon=True).start()
    return {"job_id": job_id, "status": "processing", "poll_url": f"/api/v1/jobs/{job_id}"}


@app.get("/api/v1/jobs/{job_id}", dependencies=[Depends(require_key)])
def get_job(job_id: str) -> dict[str, Any]:
    with jobs_lock:
        result = jobs.get(job_id)
    if result is None:
        raise HTTPException(status_code=404, detail="Задача не найдена или сервер был перезапущен")
    return result


@app.get("/api/v1/violation-types", dependencies=[Depends(require_key)])
def get_violation_types() -> dict[str, Any]:
    bundle = load_model() or {}
    support = bundle.get("type_video_support", {})
    trained_types = set(bundle.get("type_model").classes_.tolist()) if bundle.get("type_model") is not None else set()
    return {"legal_reviewed_on": "2026-10-18", "types": [{"type_id": key, "type": name, "trained": key in trained_types,
                       "training_videos": int(support.get(str(key), 0)),
                       "legal_qualification": LEGAL_BASES[key]} for key, name in VIOLATION_TYPES.items()]}


class TrainRequest(BaseModel):
    annotations_csv: str = "data/annotations.csv"
    threshold: float = 0.35
    event_type: str = "Возможное визуальное событие"


@app.post("/api/v1/train", dependencies=[Depends(require_key)])
def train_model(request: TrainRequest) -> dict[str, Any]:
    """Training endpoint for trusted local admin use; annotation paths are constrained to project data/."""
    csv_path = (ROOT / request.annotations_csv).resolve()
    if DATA.resolve() not in csv_path.parents:
        raise HTTPException(status_code=400, detail="Файл аннотаций должен находиться внутри data/")
    if not csv_path.is_file():
        raise HTTPException(status_code=404, detail="CSV с аннотациями не найден")
    rows = []
    with csv_path.open(encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            name = row.get("video", "").strip()
            video_path = (DATA / "videos" / name).resolve()
            if DATA.joinpath("videos").resolve() not in video_path.parents or not video_path.is_file():
                continue
            try:
                label = int(row.get("label", ""))
                timestamp = float(row["timecode_sec"]) if row.get("timecode_sec", "").strip() else None
                if label not in (0, 1):
                    continue
                if label == 1 and timestamp is None:
                    continue
                raw_type = row.get("type", "").strip()
                type_id = int(float(raw_type)) if raw_type else None
                if type_id not in VIOLATION_TYPES:
                    type_id = None
                rows.append((video_path, label, timestamp, type_id))
            except (ValueError, KeyError):
                continue
    if len(rows) < 4 or len({label for _, label, _, _ in rows}) < 2:
        raise HTTPException(status_code=422, detail="Нужно не менее 4 корректных строк и видео обоих классов (label 0 и 1)")
    X, y, groups, type_X, type_y, type_groups = [], [], [], [], [], []
    for video_path, label, timestamp, type_id in rows:
        duration, _ = video_info(video_path)
        cap = cv2.VideoCapture(str(video_path))
        if label == 1 and timestamp is not None:
            centers = [max(WINDOW_SECONDS / 2, min(duration - WINDOW_SECONDS / 2, timestamp + delta)) for delta in (-0.75, 0, 0.75)]
            centers += [max(WINDOW_SECONDS / 2, min(duration - WINDOW_SECONDS / 2, timestamp + delta)) for delta in (-8, 8)]
            labels = [1, 1, 1, 0, 0]
        else:
            centers = np.linspace(WINDOW_SECONDS / 2, max(WINDOW_SECONDS / 2, duration - WINDOW_SECONDS / 2), 6).tolist()
            labels = [label] * len(centers)
        for center, target in zip(centers, labels):
            feat = clip_feature(cap, center, duration)
            if feat is not None:
                X.append(feat)
                y.append(target)
                groups.append(training_group_key(video_path))
                if target == 1 and type_id is not None:
                    type_X.append(feat)
                    type_y.append(type_id)
                    type_groups.append(training_group_key(video_path))
        cap.release()
    if len(set(y)) < 2:
        raise HTTPException(status_code=422, detail="Не удалось получить обучающие примеры обоих классов")
    model = ExtraTreesClassifier(n_estimators=240, max_features="sqrt", min_samples_leaf=2, class_weight="balanced", random_state=42, n_jobs=max(1, (os.cpu_count() or 2) - 1))
    validation = None
    if len(set(groups)) >= 5:
        splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        train_idx, valid_idx = next(splitter.split(X, y, groups))
        if len(set(np.asarray(y)[train_idx])) == 2 and len(set(np.asarray(y)[valid_idx])) == 2:
            validation_model = ExtraTreesClassifier(n_estimators=160, max_features="sqrt", min_samples_leaf=2, class_weight="balanced", random_state=42, n_jobs=max(1, (os.cpu_count() or 2) - 1))
            validation_model.fit(np.asarray(X)[train_idx], np.asarray(y)[train_idx])
            actual = np.asarray(y)[valid_idx]
            probabilities = validation_model.predict_proba(np.asarray(X)[valid_idx])[:, 1]
            predicted = (probabilities >= float(np.clip(request.threshold, 0.1, 0.99))).astype(int)
            validation = {
                "held_out_videos": len(set(np.asarray(groups)[valid_idx])),
                "balanced_accuracy": round(float(balanced_accuracy_score(actual, predicted)), 3),
                "precision": round(float(precision_score(actual, predicted, zero_division=0)), 3),
                "recall": round(float(recall_score(actual, predicted, zero_division=0)), 3),
                "f1": round(float(f1_score(actual, predicted, zero_division=0)), 3),
                "note": "Window-level metrics on held-out videos; they do not establish event-level or different-camera performance.",
            }
    model.fit(np.asarray(X), np.asarray(y))
    type_model = None
    if type_X and len(set(type_y)) >= 2:
        type_model = ExtraTreesClassifier(n_estimators=180, max_features="sqrt", min_samples_leaf=1, class_weight="balanced", random_state=43, n_jobs=max(1, (os.cpu_count() or 2) - 1))
        type_model.fit(np.asarray(type_X), np.asarray(type_y))
    type_video_support = {str(kind): len({group for group, label_type in zip(type_groups, type_y) if label_type == kind}) for kind in sorted(set(type_y))}
    bundle = {"model": model, "type_model": type_model, "type_video_support": type_video_support,
              "threshold": float(np.clip(request.threshold, 0.1, 0.99)), "stride_sec": 0.75,
              "event_type": request.event_type, "rule": RULE, "training_rows": len(rows), "samples": len(y),
              "training_labels": {"0": int(sum(v == 0 for v in y)), "1": int(sum(v == 1 for v in y))}, "created_at": time.time()}
    joblib.dump(bundle, MODEL_PATH)
    global model_bundle
    with model_lock:
        model_bundle = bundle
    return {"status": "trained", "videos": len(rows), "windows": len(y), "model_path": "data/model.joblib", "threshold": bundle["threshold"],
            "validation": validation, "type_model_trained": type_model is not None, "type_video_support": type_video_support,
            "type_unseen": sorted(set(VIOLATION_TYPES) - set(type_y)),
            "warning": "Отложенная оценка — только ориентир: малый датасет не гарантирует качество на других ракурсах и типах событий."}


@app.get("/health")
def health() -> dict[str, Any]:
    return {"status": "ok", "model_trained": MODEL_PATH.exists(), "local_only": True}


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    types_html = "".join(f'<div class="type"><b>{type_id}.</b> {name}</div>' for type_id, name in VIOLATION_TYPES.items())
    return HTML.replace("__VIOLATION_TYPES__", types_html)


HTML = r'''<!doctype html><html lang="ru"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Forte Wither</title>
<style>*{box-sizing:border-box}body{margin:0;background:#f2f5f7;color:#15222c;font:16px system-ui,-apple-system,sans-serif}.wrap{max-width:960px;margin:42px auto;padding:0 20px}.hero{background:#102d3b;color:white;padding:30px;border-radius:18px}.hero h1{margin:0 0 8px;font-size:30px}.card{background:white;border:1px solid #dce4e8;border-radius:16px;padding:24px;margin-top:18px;box-shadow:0 4px 16px #17394c0b}.drop{border:2px dashed #9bb2bc;border-radius:12px;padding:30px;text-align:center}.btn{background:#0c6b67;color:white;border:0;border-radius:9px;padding:12px 18px;font-weight:650;cursor:pointer}.btn:disabled{opacity:.5}input[type=file]{margin:12px}.muted{color:#627782;font-size:14px}.status{margin-top:16px;font-weight:600}.event{border-top:1px solid #e2e9ec;padding:18px 0}.frames{display:flex;gap:10px;flex-wrap:wrap}.frames img{max-width:32%;border-radius:8px;border:1px solid #ddd}.tag{display:inline-block;background:#e8f4f2;color:#075c57;padding:4px 9px;border-radius:99px;font-size:13px}.error{color:#a12d23}a{color:#08746d}video{width:min(100%,640px);margin-top:16px;border-radius:12px;background:#101820}.progress{height:12px;background:#dce6e8;border-radius:99px;overflow:hidden;margin-top:14px}.progress>div{height:100%;width:0;background:linear-gradient(90deg,#0c6b67,#47c6a6);transition:width .3s}.phase{display:flex;justify-content:space-between;gap:12px}.spinner{display:inline-block;width:14px;height:14px;border:2px solid #c6d6da;border-top-color:#08746d;border-radius:50%;animation:spin .8s linear infinite;vertical-align:middle}@keyframes spin{to{transform:rotate(360deg)}}.types{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:8px}.type{background:#f7f9fa;padding:10px;border-radius:8px;font-size:14px}</style></head>
<body><main class="wrap"><section class="hero"><h1>Forte Wither</h1><div>Локальная видеоаналитика для предварительной проверки кассовых операций</div></section>
<section class="card"><h2>Проверить видео</h2><p class="muted">Выберите .mov. Сначала появится предпросмотр, затем сервис покажет ход загрузки и анализа.</p><label class="muted">API-ключ <input id="key" type="password" placeholder="Введите ключ из .api_key" autocomplete="off"></label><div class="drop"><input id="file" type="file" accept=".mov,video/quicktime"><br><button class="btn" id="go">Начать анализ</button><br><video id="preview" controls playsinline hidden></video></div><div class="phase"><span id="status" class="status">Ожидает видео</span><span id="percent" class="muted">0%</span></div><div class="progress"><div id="bar"></div></div><div id="result"></div></section>
<section class="card"><h2>Типы событий</h2><p class="muted">Модель указывает наиболее вероятную категорию и статус правовой проверки. Видеофрагмент сам по себе не устанавливает факт нарушения.</p><div class="types">__VIOLATION_TYPES__</div><p class="muted">Правовая база проверена по ИПС «Әділет» 18.10.2026; учтены изменения к постановлению № 231, вступившие в силу 20.07.2026. Для части категорий требуется сверка с внутренними правилами банка. Типы 2, 10 и 11 пока не представлены в размеченных обучающих видео.</p></section>
<section class="card"><h3>Правовая основа</h3><p><a href="https://adilet.zan.kz/rus/docs/V1900019680" target="_blank" rel="noreferrer">Действующие правила кассовых операций и инкассации — постановление НБРК № 231 от 29.11.2019</a></p><p><a href="https://adilet.zan.kz/rus/docs/V2000020075" target="_blank" rel="noreferrer">Правила организации охраны и устройства помещений банков — постановление НБРК № 14 от 24.02.2020</a></p><p><a href="https://adilet.zan.kz/rus/docs/V2400034348" target="_blank" rel="noreferrer">Инструкция по антитеррористической защите уязвимых расчетно-кассовых отделений банков</a></p><p class="muted">Сервис сообщает о визуальных признаках и возможной применимости нормы. Итоговую квалификацию уполномоченный сотрудник делает после проверки видео, документов, полномочий и внутренних актов банка.</p></section></main>
<script>const $=s=>document.querySelector(s);let previewUrl=null;$('#file').onchange=()=>{const f=$('#file').files[0],v=$('#preview');if(previewUrl)URL.revokeObjectURL(previewUrl);if(f){previewUrl=URL.createObjectURL(f);v.src=previewUrl;v.hidden=false;$('#status').textContent='Видео выбрано — готово к анализу'}else v.hidden=true};function setProgress(n,msg,active=true){n=Math.max(0,Math.min(100,n));$('#bar').style.width=n+'%';$('#percent').textContent=Math.round(n)+'%';$('#status').innerHTML=(active&&n<100?'<i class="spinner"></i> ':'')+esc(msg)}function upload(fd,key){return new Promise((resolve,reject)=>{const x=new XMLHttpRequest();x.open('POST','/api/v1/analyze');x.setRequestHeader('X-API-Key',key);x.upload.onprogress=e=>{if(e.lengthComputable)setProgress(e.loaded/e.total*20,'Загрузка видео')};x.onerror=()=>reject(Error('Ошибка сети при загрузке'));x.onload=()=>{let d={};try{d=JSON.parse(x.responseText)}catch{};if(x.status>=200&&x.status<300)resolve(d);else reject(Error(d.detail||'Ошибка загрузки'))};x.send(fd)})}$('#go').onclick=async()=>{const f=$('#file').files[0],key=$('#key').value;if(!key){setProgress(0,'Введите API-ключ',false);return}if(!f){setProgress(0,'Выберите .mov файл',false);return}$('#go').disabled=true;$('#result').innerHTML='';try{const fd=new FormData();fd.append('file',f);setProgress(0,'Загрузка видео');const {job_id}=await upload(fd,key);let done=false;while(!done){await new Promise(x=>setTimeout(x,1200));const p=await fetch('/api/v1/jobs/'+job_id,{headers:{'X-API-Key':key}});if(!p.ok)throw Error('Не удалось получить статус обработки');const d=await p.json();if(d.status==='failed')throw Error(d.error);setProgress(20+(d.progress_percent||0)*.8,d.progress_message||'Анализ кадров');if(d.status==='done'){done=true;setProgress(100,'Обработка завершена',false);render(d)}}}catch(e){setProgress(0,e.message,false);$('#status').classList.add('error')}finally{$('#go').disabled=false}};function render(d){$('#status').textContent=d.verdict==='violation'?'Обнаружены признаки возможных нарушений':'Нарушение не обнаружено';$('#result').innerHTML=(d.violations||[]).map(v=>'<article class="event"><span class="tag">Тип '+(v.type_id||'—')+': '+esc(v.type)+'</span> <strong>Таймкод: '+Number(v.timecode_sec).toFixed(2)+' сек.</strong>'+(v.type_confidence!==null?'<p class="muted">Уверенность в типе: '+Math.round(v.type_confidence*100)+'%</p>':'')+'<p>'+esc(v.description)+'</p><div class="frames">'+(v.evidence_frames||[]).map(f=>'<figure><img src="data:image/jpeg;base64,'+f.image_base64+'"><figcaption class="muted">'+f.timestamp_sec+' сек.</figcaption></figure>').join('')+'</div>'+(v.legal_qualification?'<p><strong>'+esc(v.legal_qualification.label)+'</strong></p><p class="muted">'+esc(v.legal_qualification.document)+' · '+esc(v.legal_qualification.paragraphs)+'<br>'+esc(v.legal_qualification.note)+'<br><a href="'+v.legal_qualification.url+'" target="_blank" rel="noreferrer">Открыть НПА в ИПС «Әділет»</a></p>':'')+'</article>').join('')||'<p class="muted">Событий для проверки модель не выделила.</p>'}function esc(s){return String(s||'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]))}</script></body></html>'''

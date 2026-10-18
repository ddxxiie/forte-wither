import argparse
import json

from app import TrainRequest, train_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Обучить локальную модель кассовых видео")
    parser.add_argument("--annotations", default="data/annotations.csv")
    parser.add_argument("--threshold", type=float, default=0.35)
    parser.add_argument("--event-type", default="Возможное визуальное событие")
    args = parser.parse_args()
    result = train_model(TrainRequest(annotations_csv=args.annotations, threshold=args.threshold, event_type=args.event_type))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

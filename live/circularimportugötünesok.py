# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import argparse
import cv2
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
from ultralytics.utils.files import increment_path


class SAHIInference:
    def __init__(self):
        self.detection_model = None

    def load_model(self, weights: str = "bestmodels/best13.pt", device: str = "cuda") -> None:
        from ultralytics.utils.torch_utils import select_device
        from sahi import AutoDetectionModel
        import os

        yolo11_model_path = "bestmodels/best13.pt"

        if not os.path.exists(yolo11_model_path):
            raise FileNotFoundError(f"Model weights not found at: {yolo11_model_path}")

        self.detection_model = AutoDetectionModel.from_pretrained(
            model_type="ultralytics",
            model_path=yolo11_model_path,
            confidence_threshold=0.7,
            device=select_device(device)
        )

        print(f"Model loaded successfully from: {yolo11_model_path}")

    def inference(
        self,
        weights: str = "bestmodels/best13.pt",
        source: str = "BeforeSplit/DJI_20251028133726_0001_V.MP4",
        view_img: bool = False,
        save_img: bool = False,
        exist_ok: bool = False,
        device: str = "",
        hide_conf: bool = False,
        slice_width: int = 512,
        slice_height: int = 512,
    ) -> None:
        cap = cv2.VideoCapture(source)
        assert cap.isOpened(), f"❌ Error reading video file: {source}"

        save_dir = increment_path("runs/detect/predict", exist_ok)
        save_dir.mkdir(parents=True, exist_ok=True)

        self.load_model(weights, device)
        idx = 0

        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break

            results = get_sliced_prediction(
                frame[..., ::-1],
                self.detection_model,
                slice_height=slice_height,
                slice_width=slice_width,
            )

            if view_img:
                cv2.imshow("Ultralytics YOLO Inference", frame)

            if save_img:
                idx += 1
                results.export_visuals(export_dir=save_dir, file_name=f"img_{idx}", hide_conf=hide_conf)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            
            cv2.imshow('rrrrrr',frame)
        cap.release()
        cv2.destroyAllWindows()

    @staticmethod
    def parse_opt() -> argparse.Namespace:
        parser = argparse.ArgumentParser()
        parser.add_argument("--view-img", action="store_true", help="show results")
        parser.add_argument("--save-img", action="store_true", help="save results")
        parser.add_argument("--exist-ok", action="store_true", help="existing project/name ok, do not increment")
        parser.add_argument("--device", default="", help="cuda device, i.e. 0 or 0,1,2,3 or cpu")
        parser.add_argument("--hide-conf", default=False, action="store_true", help="display or hide confidences")
        parser.add_argument("--slice-width", default=512, type=int, help="Slice width for inference")
        parser.add_argument("--slice-height", default=512, type=int, help="Slice height for inference")
        return parser.parse_args()


if __name__ == "__main__":
    inference = SAHIInference()
    inference.inference(**vars(inference.parse_opt()))
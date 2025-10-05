from ultralytics import YOLO
import cv2, time, argparse, os

def main():
    print("[DEBUG] starting predict_video")
    ap = argparse.ArgumentParser(description="Run YOLO detection on a video and save an annotated video.")
    ap.add_argument("--model", default="yolov8n.pt", help="YOLO weights (auto-downloads if needed).")
    ap.add_argument("--source", default="assets/sample_drive.mp4", help="Path to input video file.")
    ap.add_argument("--out", default="runs/predict/demo_det.mp4", help="Path to output video file.")
    ap.add_argument("--conf", type=float, default=0.25, help="Confidence threshold.")
    ap.add_argument("--imgsz", type=int, default=640, help="Inference image size.")
    ap.add_argument("--device", default=None, help="mps | cpu | 0 (CUDA)")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    print("[DEBUG] loading model:", args.model)
    model = YOLO(args.model)

    print("[DEBUG] opening video:", args.source)
    cap = cv2.VideoCapture(args.source)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open video: {args.source}")
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_in = cap.get(cv2.CAP_PROP_FPS) or 30.0

    print(f"[DEBUG] input size={w}x{h} fps={fps_in}")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(args.out, fourcc, fps_in, (w, h))
    if not writer.isOpened():
        raise SystemExit(f"Cannot open writer for: {args.out}. Try .avi and FourCC XVID.")

    t0 = time.time(); frames = 0
    while True:
        ok, frame = cap.read()
        if not ok: break
        results = model.predict(source=frame, conf=args.conf, imgsz=args.imgsz, device=args.device, verbose=False)
        annotated = results[0].plot()
        writer.write(annotated)
        frames += 1

    cap.release(); writer.release()
    dt = time.time() - t0
    print(f"Processed {frames} frames in {dt:.2f}s => {frames/max(dt,1e-6):.2f} FPS")
    print("Saved:", args.out)

if __name__ == "__main__":
    main()

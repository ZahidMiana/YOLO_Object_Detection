"""
YOLOv8 Object Detection - Hugging Face Spaces
Image Upload + Real-Time Webcam Streaming Detection
"""

# ---- FIX: Patch gradio_client bug before importing gradio ----
import utils.session
import gradio_client.utils as _gc_utils

_orig_get_type = _gc_utils.get_type
def _safe_get_type(schema):
    if not isinstance(schema, dict):
        return "Any"
    return _orig_get_type(schema)
_gc_utils.get_type = _safe_get_type

_orig_jstpt = _gc_utils._json_schema_to_python_type
def _safe_jstpt(schema, defs=None):
    if isinstance(schema, bool):
        return "Any"
    return _orig_jstpt(schema, defs)
_gc_utils._json_schema_to_python_type = _safe_jstpt
# ---- END FIX ----

import gradio as gr
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO

model = YOLO("yolov8s.pt")

COLORS = {}
np.random.seed(42)
for i in range(80):
    COLORS[i] = tuple(np.random.randint(50, 255, size=3).tolist())

def draw_boxes(image, boxes, scores, class_ids, names, line_width=2):
    img = image.copy()
    for box, score, cid in zip(boxes, scores, class_ids):
        x1, y1, x2, y2 = map(int, box)
        color = COLORS.get(cid, (0, 255, 0))
        label = f"{names[cid]} {score:.2f}"
        cv2.rectangle(img, (x1, y1), (x2, y2), color, line_width)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(img, (x1, max(y1 - th - 8, 0)), (x1 + tw + 4, y1), color, -1)
        cv2.putText(img, label, (x1 + 2, max(y1 - 4, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return img

def detect_image(image, conf_threshold, iou_threshold):
    if image is None:
        return None, "Upload an image first"
    img_array = np.array(image)
    results = model(img_array, conf=conf_threshold, iou=iou_threshold, verbose=False)
    result = results[0]
    boxes = result.boxes
    if len(boxes) == 0:
        return Image.fromarray(img_array), "No objects detected"
    bbox = boxes.xyxy.cpu().numpy()
    confs = boxes.conf.cpu().numpy()
    cls_ids = boxes.cls.cpu().numpy().astype(int)
    annotated = draw_boxes(img_array, bbox, confs, cls_ids, model.names, line_width=3)
    lines = []
    for i in range(len(confs)):
        x1, y1, x2, y2 = bbox[i].astype(int)
        name = model.names[cls_ids[i]]
        lines.append(f"{i+1}. {name} - {confs[i]*100:.1f}% - [{x1},{y1},{x2},{y2}]")
    classes_found = list(set([model.names[c] for c in cls_ids]))
    summary = (f"Detected {len(confs)} object(s): {', '.join(classes_found)}\n"
               f"Avg Confidence: {confs.mean()*100:.1f}% | "
               f"Inference: {result.speed['inference']:.0f}ms\n"
               + "-" * 40 + "\n" + "\n".join(lines))
    return Image.fromarray(annotated), summary

def detect_webcam_frame(frame, conf_threshold):
    if frame is None:
        return frame
    img_array = np.array(frame)
    results = model(img_array, conf=conf_threshold, iou=0.45, verbose=False)
    result = results[0]
    boxes = result.boxes
    if len(boxes) == 0:
        cv2.putText(img_array, "No objects", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        return Image.fromarray(img_array)
    bbox = boxes.xyxy.cpu().numpy()
    confs = boxes.conf.cpu().numpy()
    cls_ids = boxes.cls.cpu().numpy().astype(int)
    annotated = draw_boxes(img_array, bbox, confs, cls_ids, model.names, line_width=2)
    n = len(confs)
    classes_found = list(set([model.names[c] for c in cls_ids]))
    info = f"{n} objects: {', '.join(classes_found)} | Avg: {confs.mean()*100:.0f}%"
    cv2.putText(annotated, info, (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
    return Image.fromarray(annotated)

css = """
h1 { text-align:center!important; color:#764ba2; }
.contain { max-width: 1200px; margin: auto; }
"""

with gr.Blocks(css=css, theme=gr.themes.Soft(), title="YOLOv8 Detection") as demo:
    gr.Markdown("# YOLOv8 Object Detection\n"
                "Image upload karo ya camera se real-time detect karo — "
                "CNN based, 80 COCO classes, bounding boxes + confidence scores")

    with gr.Tabs():
        with gr.TabItem("Image Detection"):
            with gr.Row():
                with gr.Column():
                    img_input = gr.Image(type="pil", label="Upload Image", height=400)
                    img_conf = gr.Slider(0.1, 1.0, value=0.25, step=0.05,
                                         label="Confidence Threshold")
                    img_iou = gr.Slider(0.1, 1.0, value=0.45, step=0.05,
                                        label="IoU Threshold (NMS)")
                    img_btn = gr.Button("Detect Objects", variant="primary", size="lg")
                with gr.Column():
                    img_output = gr.Image(type="pil", label="Detection Results", height=400)
                    img_details = gr.Textbox(label="Detection Details", lines=6,
                                              interactive=False)

            img_btn.click(detect_image,
                          inputs=[img_input, img_conf, img_iou],
                          outputs=[img_output, img_details])

        with gr.TabItem("Real-Time Camera"):
            gr.Markdown("**Camera on karo** — har frame pe YOLO detection real-time chalega. "
                        "Bounding boxes aur confidence scores live dikhein ge.")
            with gr.Row():
                cam_conf = gr.Slider(0.15, 1.0, value=0.30, step=0.05,
                                     label="Confidence Threshold")
            with gr.Row():
                webcam_in = gr.Image(sources=["webcam"], type="pil",
                                     label="Camera Feed", streaming=True, height=480)
                webcam_out = gr.Image(type="pil", label="Live Detection", height=480)

            webcam_in.stream(detect_webcam_frame,
                             inputs=[webcam_in, cam_conf],
                             outputs=[webcam_out],
                             stream_every=0.15)

    gr.Markdown("<center>YOLOv8 + Gradio | CNN Object Detection | 80 COCO Classes</center>")

if __name__ == "__main__":
    demo.launch()

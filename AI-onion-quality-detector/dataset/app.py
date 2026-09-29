import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

from flask import Flask, render_template, request
import numpy as np
import cv2
import onnxruntime as ort
from werkzeug.utils import secure_filename
from pathlib import Path
import uuid


app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent

UPLOAD_FOLDER = BASE_DIR / "static" / "uploads"
RESULT_FOLDER = BASE_DIR / "static" / "results"

UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)
RESULT_FOLDER.mkdir(parents=True, exist_ok=True)


# 3-CLASS TRAINED ONNX MODEL
MODEL_PATH = BASE_DIR.parent / "model" / "best.onnx"

session = ort.InferenceSession(
    str(MODEL_PATH),
    providers=["CPUExecutionProvider"]
)

input_name = session.get_inputs()[0].name


@app.route("/", methods=["GET", "POST"])
def home():

    result = None
    image_path = None
    detections = []

    fresh = 0
    defected = 0
    non_onion = 0
    uncertain = 0

    total_onions = 0

    fresh_percentage = 0
    defected_percentage = 0

    if request.method == "POST":

        # Check uploaded file
        if "image" not in request.files:
            return render_template(
                "index.html",
                error="Please select an image."
            )

        file = request.files["image"]

        if file.filename == "":
            return render_template(
                "index.html",
                error="Please select an image."
            )

        # Save uploaded image
        filename = secure_filename(file.filename)

        unique_name = f"{uuid.uuid4().hex}_{filename}"

        upload_path = UPLOAD_FOLDER / unique_name

        file.save(upload_path)


        # =========================
        # READ IMAGE
        # =========================

        image = cv2.imread(str(upload_path))

        if image is None:
            return render_template(
                "index.html",
                error="Unable to read the uploaded image."
            )

        original = image.copy()


        # =========================
        # PREPROCESS IMAGE
        # =========================

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        resized = cv2.resize(
            image_rgb,
            (320, 320)
        )

        input_image = resized.astype(
            np.float32
        ) / 255.0

        input_image = np.transpose(
            input_image,
            (2, 0, 1)
        )

        input_image = np.expand_dims(
            input_image,
            axis=0
        )


        # =========================
        # ONNX INFERENCE
        # =========================

        outputs = session.run(
            None,
            {input_name: input_image}
        )

        predictions = outputs[0][0]

        # YOLO output:
        # 7 x 8400
        predictions = predictions.T


        # =========================
        # IMAGE INFORMATION
        # =========================

        result_name = f"result_{unique_name}"

        result_path = RESULT_FOLDER / result_name

        image_path = f"/static/results/{result_name}"


        # =========================
        # PARSE DETECTIONS
        # =========================

        orig_h, orig_w = original.shape[:2]

        scale_x = orig_w / 320
        scale_y = orig_h / 320

        boxes = []
        scores = []
        class_ids = []


        for row in predictions:

            x, y, w, h = row[:4]

            class_scores = row[4:]

            class_id = int(
                np.argmax(class_scores)
            )

            confidence = float(
                class_scores[class_id]
            )

            # Ignore very low confidence
            if confidence < 0.25:
                continue


            # Convert center coordinates
            # to corner coordinates

            x1 = int(
                (x - w / 2) * scale_x
            )

            y1 = int(
                (y - h / 2) * scale_y
            )

            x2 = int(
                (x + w / 2) * scale_x
            )

            y2 = int(
                (y + h / 2) * scale_y
            )


            # Keep coordinates inside image

            x1 = max(
                0,
                min(x1, orig_w - 1)
            )

            y1 = max(
                0,
                min(y1, orig_h - 1)
            )

            x2 = max(
                0,
                min(x2, orig_w - 1)
            )

            y2 = max(
                0,
                min(y2, orig_h - 1)
            )


            boxes.append([
                x1,
                y1,
                x2 - x1,
                y2 - y1
            ])

            scores.append(confidence)

            class_ids.append(class_id)


        # =========================
        # NON-MAXIMUM SUPPRESSION
        # =========================

        selected = cv2.dnn.NMSBoxes(
            boxes,
            scores,
            score_threshold=0.25,
            nms_threshold=0.45
        )


        if len(selected) == 0:
            selected = []


        # =========================
        # CLASS NAMES
        # =========================

        class_names = {
            0: "Fresh Onion",
            1: "Defected Onion",
            2: "Non-Onion"
        }


        # =========================
        # PROCESS DETECTIONS
        # =========================

        for index in selected:

            index = int(index)

            x, y, w, h = boxes[index]

            confidence = scores[index]

            class_id = class_ids[index]

            class_name = class_names.get(
                class_id,
                "Unknown"
            )


            confidence_percent = round(
                confidence * 100,
                2
            )


            # =========================
            # CONFIDENCE DECISION
            # =========================

            if confidence >= 0.70:
                status = "Confirmed"
            else:
                status = "Needs Verification"


            # =========================
            # PHYSICAL MARKING
            # =========================

            if status == "Needs Verification":

                mark = "YELLOW"

            elif class_name == "Fresh Onion":

                mark = "GREEN"

            elif class_name == "Defected Onion":

                mark = "RED"

            else:

                mark = "GRAY"


            # =========================
            # STORE DETECTION
            # =========================

            detections.append({
                "name": class_name,
                "confidence": confidence_percent,
                "status": status,
                "mark": mark
            })


            # =========================
            # DRAW BOUNDING BOX
            # =========================

            cv2.rectangle(
                original,
                (x, y),
                (x + w, y + h),
                (0, 255, 0),
                2
            )


            label = (
                f"{class_name} "
                f"{confidence_percent}%"
            )


            cv2.putText(
                original,
                label,
                (x, max(20, y - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 0),
                2
            )


            # =========================
            # COUNT RESULTS
            # =========================

            if status == "Confirmed":

                if class_name == "Fresh Onion":

                    fresh += 1

                elif class_name == "Defected Onion":

                    defected += 1

                elif class_name == "Non-Onion":

                    non_onion += 1

            else:

                uncertain += 1


        # =========================
        # SAVE RESULT IMAGE
        # =========================

        cv2.imwrite(
            str(result_path),
            original
        )


        # =========================
        # ONION PERCENTAGES
        # =========================

        total_onions = fresh + defected


        if total_onions > 0:

            fresh_percentage = round(
                (fresh / total_onions) * 100,
                2
            )

            defected_percentage = round(
                (defected / total_onions) * 100,
                2
            )


        # =========================
        # RESULT MESSAGE
        # =========================

        result = {
            "name": "Analysis Complete",
            "total": len(detections)
        }


    # =========================
    # RENDER WEBSITE
    # =========================

    return render_template(
        "index.html",

        result=result,

        image_path=image_path,

        detections=detections,

        fresh=fresh,

        defected=defected,

        non_onion=non_onion,

        uncertain=uncertain,

        total_onions=total_onions,

        fresh_percentage=fresh_percentage,

        defected_percentage=defected_percentage
    )


if __name__ == "__main__":
    app.run(
        debug=True
    )
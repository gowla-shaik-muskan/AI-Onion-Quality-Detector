from flask import Flask, render_template, request
from ultralytics import YOLO
import os
from werkzeug.utils import secure_filename
from pathlib import Path
import uuid

app = Flask(
    __name__,
    template_folder=os.path.join(os.path.dirname(__file__), "templates")
)

BASE_DIR = Path(__file__).resolve().parent

UPLOAD_FOLDER = BASE_DIR / "static" / "uploads"
RESULT_FOLDER = BASE_DIR / "static" / "results"

UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)
RESULT_FOLDER.mkdir(parents=True, exist_ok=True)

# 3-CLASS TRAINED MODEL
model = YOLO(
    r"D:\AI-Onion-Quality-Detector\runs\detect\onion_3class_test\weights\best.pt"
)


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

        filename = secure_filename(file.filename)

        unique_name = f"{uuid.uuid4().hex}_{filename}"

        upload_path = UPLOAD_FOLDER / unique_name

        file.save(upload_path)

        # YOLO prediction
        results = model.predict(
            source=str(upload_path),
            conf=0.25,
            verbose=False
        )

        prediction = results[0]

        # Save annotated image
        result_name = f"result_{unique_name}"
        result_path = RESULT_FOLDER / result_name

        prediction.save(filename=str(result_path))

        image_path = f"/static/results/{result_name}"

        # Process detections
        for box in prediction.boxes:

            class_id = int(box.cls[0])
            confidence = float(box.conf[0])

            class_name = model.names[class_id]

            confidence_percent = round(
                confidence * 100,
                2
            )

            # Confidence status
            if confidence >= 0.70:
                status = "Confirmed"
            else:
                status = "Needs Verification"

            # Physical marking concept
            if status == "Needs Verification":
                mark = "YELLOW"

            elif class_name == "Fresh Onion":
                mark = "GREEN"

            elif class_name == "Defected Onion":
                mark = "RED"

            else:
                mark = "GRAY"

            detections.append({
                "name": class_name,
                "confidence": confidence_percent,
                "status": status,
                "mark": mark
            })

            # Count detections
            if confidence >= 0.70:

                if class_name == "Fresh Onion":
                    fresh += 1

                elif class_name == "Defected Onion":
                    defected += 1

                elif class_name == "Non-Onion":
                    non_onion += 1

            else:
                uncertain += 1

        # Only onions are used for onion-quality percentages
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

        result = {
            "name": "Analysis Complete",
            "total": len(detections)
        }

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
    app.run(debug=True)
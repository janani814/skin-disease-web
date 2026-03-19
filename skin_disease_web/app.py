from flask import Flask, render_template, request, send_file
from ultralytics import YOLO
import os
import base64
from reportlab.pdfgen import canvas

app = Flask(__name__)

model = YOLO("best.pt")

UPLOAD_FOLDER = "static/uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Disease full names
disease_names = {
    "akiec": "Actinic Keratosis",
    "bcc": "Basal Cell Carcinoma",
    "bkl": "Benign Keratosis",
    "df": "Dermatofibroma",
    "mel": "Melanoma",
    "nv": "Melanocytic Nevus",
    "vasc": "Vascular Lesion"
}


# Extra info
disease_info = {

    "Actinic Keratosis": {
        "treatment": "Apply aloe vera and turmeric paste.",
        "doctor": "Dr. Priya Raman",
        "hospital": "Kauvery Hospital, Trichy",
        "advantage": "Reduces skin inflammation naturally.",
        "child_safe": "Not Required"
    },

    "Basal Cell Carcinoma": {
        "treatment": "Use neem oil and turmeric paste.",
        "doctor": "Dr. Suresh Kumar",
        "hospital": "Apollo Hospital, Chennai",
        "advantage": "Controls infection and irritation.",
        "child_safe": "Doctor Consultation Required"
    },

    "Benign Keratosis": {
        "treatment": "Apply coconut oil and sandalwood.",
        "doctor": "Dr. Meena Lakshmi",
        "hospital": "CMC Hospital, Vellore",
        "advantage": "Improves skin texture.",
        "child_safe": "Safe"
    },

    "Dermatofibroma": {
        "treatment": "Use turmeric and honey paste.",
        "doctor": "Dr. Arjun Patel",
        "hospital": "AIIMS, Delhi",
        "advantage": "Helps skin healing.",
        "child_safe": "Safe"
    },

    "Melanoma": {
        "treatment": "Immediate medical consultation required.",
        "doctor": "Dr. Rahul Menon",
        "hospital": "Tata Memorial Hospital, Mumbai",
        "advantage": "Early detection saves life.",
        "child_safe": "Doctor Consultation Required"
    },

    "Melanocytic Nevus": {
        "treatment": "Apply aloe vera gel.",
        "doctor": "Dr. Kavitha Reddy",
        "hospital": "Fortis Hospital, Bangalore",
        "advantage": "Keeps skin moisturized.",
        "child_safe": "Safe"
    },

    "Vascular Lesion": {
        "treatment": "Use neem paste.",
        "doctor": "Dr. Vivek Sharma",
        "hospital": "Manipal Hospital, Bangalore",
        "advantage": "Improves blood circulation.",
        "child_safe": "Doctor Consultation Required"
    }
}

# store result globally
last_result = {}


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    name = request.form["name"]
    age = request.form["age"]

    image_data = request.form.get("image_data")

    if image_data:
        image_data = image_data.split(",")[1]
        image_bytes = base64.b64decode(image_data)

        filepath = os.path.join(app.config["UPLOAD_FOLDER"], "capture.png")

        with open(filepath, "wb") as f:
            f.write(image_bytes)

    else:
        file = request.files["image"]
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
        file.save(filepath)

    results = model(filepath)

    probs = results[0].probs.data.tolist()
    class_names = results[0].names

    predicted_index = probs.index(max(probs))
    predicted_class = class_names[predicted_index]

    disease = disease_names.get(predicted_class, predicted_class)

    confidence = round(max(probs) * 100, 2)

    info = disease_info[disease]

    # save for PDF
    global last_result
    last_result = {
        "name": name,
        "age": age,
        "disease": disease,
        "confidence": confidence,
        "treatment": info["treatment"],
        "doctor": info["doctor"],
        "hospital": info["hospital"]
    }

    return render_template(
        "result.html",
        name=name,
        age=age,
        disease=disease,
        confidence=confidence,
        treatment=info["treatment"],
        doctor=info["doctor"],
        hospital=info["hospital"],
        advantage=info["advantage"],
        child_safe=info["child_safe"],
        image=filepath
    )


@app.route("/download_pdf")
def download_pdf():

    file_path = "report.pdf"

    c = canvas.Canvas(file_path)

    c.drawString(100, 750, "Skin Disease Detection Report")

    c.drawString(100, 720, "Name: " + last_result["name"])
    c.drawString(100, 700, "Age: " + last_result["age"])
    c.drawString(100, 680, "Disease: " + last_result["disease"])
    c.drawString(100, 660, "Confidence: " + str(last_result["confidence"]) + "%")

    c.drawString(100, 630, "Treatment: " + last_result["treatment"])
    c.drawString(100, 610, "Doctor: " + last_result["doctor"])
    c.drawString(100, 590, "Hospital: " + last_result["hospital"])

    c.save()

    return send_file(file_path, as_attachment=True)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)
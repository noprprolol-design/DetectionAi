from flask import Flask, render_template, request, jsonify
from ultralytics import YOLO
import cv2
import numpy as np
import base64
import time
import os


# ===============================
# SETUP FLASK
# ===============================

app = Flask(__name__)


# ===============================
# FOLDER PENYIMPANAN
# ===============================

os.makedirs(
    "static/upload",
    exist_ok=True
)

os.makedirs(
    "static/result",
    exist_ok=True
)


# ===============================
# LOAD MODEL YOLO
# ===============================

model = YOLO("model/best.pt")


# ===============================
# WARNA BOUNDING BOX
# ===============================

COLORS = {

    "kopi": (0, 165, 255),

    "teh": (0, 255, 0),

    "susu": (255, 0, 0)

}


# ===============================
# HOME PAGE
# ===============================

@app.route("/")
def home():

    return render_template(
        "index.html"
    )



# ===============================
# API DETEKSI GAMBAR
# ===============================

@app.route(
    "/api/detect",
    methods=["POST"]
)

def detect():


    start = time.time()


    # cek file

    if "file" not in request.files:


        return jsonify({

            "success": False,

            "error":
            "Gambar tidak ditemukan"

        })



    file = request.files["file"]



    # ===============================
    # SIMPAN GAMBAR ASLI
    # ===============================


    upload_path = (

        "static/upload/"

        + file.filename

    )


    file.save(

        upload_path

    )



    # baca gambar


    img = cv2.imread(

        upload_path

    )



    # ===============================
    # YOLO DETECTION
    # ===============================


    results = model(

        img,

        conf=0.5

    )



    detections = []



    for result in results:



        for box in result.boxes:



            # koordinat box


            x1, y1, x2, y2 = map(

                int,

                box.xyxy[0]

            )



            # class id


            cls = int(

                box.cls[0]

            )



            # confidence


            conf = float(

                box.conf[0]

            )



            # nama class


            name = model.names[cls]



            detections.append({

                "class": name,

                "confidence": round(

                    conf * 100,

                    2

                )

            })



            color = COLORS.get(

                name,

                (255,255,255)

            )



            # ===============================
            # GAMBAR BOUNDING BOX
            # ===============================


            cv2.rectangle(

                img,

                (x1,y1),

                (x2,y2),

                color,

                3

            )



            label = (

                f"{name} "

                f"{conf*100:.1f}%"

            )



            cv2.putText(

                img,

                label,

                (x1,y1-10),

                cv2.FONT_HERSHEY_SIMPLEX,

                0.8,

                color,

                2

            )




    # ===============================
    # SIMPAN HASIL DETEKSI
    # ===============================


    result_path = (

        "static/result/result_"

        + file.filename

    )


    cv2.imwrite(

        result_path,

        img

    )



    # ===============================
    # CONVERT KE BASE64 UNTUK WEB
    # ===============================


    _, buffer = cv2.imencode(

        ".jpg",

        img

    )



    image_base64 = base64.b64encode(

        buffer

    ).decode(

        "utf-8"

    )



    end = time.time()



    # ===============================
    # KIRIM KE FRONTEND
    # ===============================


    return jsonify({

        "success": True,


        "image": image_base64,


        "detections": detections,


        "total": len(

            detections

        ),


        "inference_time": round(

            (end-start)*1000

        )

    })




# ===============================
# RUN SERVER
# ===============================

if __name__ == "__main__":


    app.run(

        debug=True

    )
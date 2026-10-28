"""
app.py

Single Flask entry point for Book Analyzer.

This replaces the old top-level analyzer.py — that file duplicated this
same routing logic and is no longer needed; delete it.
"""

import os
from flask import Flask, render_template, request
from pypdf import PdfReader
from werkzeug.utils import secure_filename

# Import analysis functions from analyzer package
from analyzer.nlp_engine import analyze_chapters, analyze_text

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 25 * 1024 * 1024  # 25 MB upload limit

ALLOWED_EXTENSIONS = {".pdf"}


def _is_pdf(filename):
    return os.path.splitext(filename)[1].lower() in ALLOWED_EXTENSIONS


def _extract_text(file_stream):
    reader = PdfReader(file_stream)
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n".join(pages)


@app.route("/", methods=["GET", "POST"])
def upload_file():
    if request.method == "GET":
        # Show colorful upload page
        return render_template("upload.html")

    file = request.files.get("file")

    if not file or file.filename == "":
        return render_template("upload.html", error="Please choose a PDF file to upload.")

    if not _is_pdf(file.filename):
        return render_template("upload.html", error="Only PDF files are supported.")

    filename = secure_filename(file.filename)

    try:
        text = _extract_text(file.stream)
    except Exception:
        return render_template(
            "upload.html",
            error="Couldn't read that PDF — it may be corrupted or password-protected.",
        )

    if not text.strip():
        return render_template(
            "upload.html",
            error="No extractable text was found in that PDF (it may be a scanned/image-only book).",
        )

    # Run analysis
    analysis = analyze_text(text)
    chapters = analyze_chapters(text)

    # Show colorful dashboard
    return render_template(
        "report.html",
        filename=filename,
        summary=analysis["summary"],
        keywords=analysis["keywords"],
        quotes=analysis["quotes"],
        readability=analysis["readability"],
        short_description=analysis["short_description"],
        genres=analysis["genres"],
        chapters=chapters,
    )


@app.errorhandler(413)
def file_too_large(_error):
    return render_template("upload.html", error="That file is too large (25 MB limit)."), 413


if __name__ == "__main__":
    app.run(debug=True)

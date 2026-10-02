"""HTTP API for the four prediction pipelines, plus the built interface.

One Cloud Run service serves both: the Next.js static export is mounted at the
root and the API lives under /api, which removes the need for CORS and halves
the deployment surface.

Files are uploaded one at a time rather than as a batch, because Cloud Run caps
a request at 32 MB and a single rail recording is already 17 MB. It also lets
the interface show honest per-file progress instead of one long stall.
"""
import os
import sys
import tempfile
import traceback

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import FileResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "src"))
sys.path.insert(0, HERE)

import service  # noqa: E402

MAX_BYTES = 40 * 1024 * 1024
STATIC = os.path.join(HERE, "static")

app = FastAPI(title="Living Railway", docs_url="/api/docs")
app.add_middleware(GZipMiddleware, minimum_size=1000)


@app.get("/api/health")
def health():
    return {"ok": True, "subsystems": sorted(service.SUBSYSTEMS)}


@app.get("/api/subsystems")
def subsystems():
    return [dict(id=k, **v) for k, v in service.SUBSYSTEMS.items()]


@app.post("/api/predict/{subsystem}")
async def predict(subsystem: str, file: UploadFile = File(...)):
    if subsystem not in service.SUBSYSTEMS:
        raise HTTPException(404, "Unknown subsystem %r" % subsystem)

    accepts = service.SUBSYSTEMS[subsystem]["accepts"]
    if not file.filename.lower().endswith(accepts):
        raise HTTPException(
            400, "%s expects a %s file, but %s was uploaded."
                 % (service.SUBSYSTEMS[subsystem]["label"], accepts, file.filename))

    data = await file.read()
    if len(data) > MAX_BYTES:
        raise HTTPException(413, "File is %.1f MB; the limit is %d MB."
                            % (len(data) / 1e6, MAX_BYTES // 1024 // 1024))
    if not data:
        raise HTTPException(400, "That file is empty.")

    suffix = os.path.splitext(file.filename)[1] or ".dat"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    try:
        tmp.write(data)
        tmp.close()
        return service.predict(subsystem, tmp.name, file.filename)
    except HTTPException:
        raise
    except Exception as exc:
        traceback.print_exc()
        raise HTTPException(
            422, "Could not read %s as a %s file. %s"
                 % (file.filename, service.SUBSYSTEMS[subsystem]["label"], exc))
    finally:
        try:
            os.unlink(tmp.name)
        except OSError:
            pass


class ExportRequest(BaseModel):
    rows: list[dict]


@app.post("/api/export/{subsystem}")
def export(subsystem: str, req: ExportRequest):
    if subsystem not in service.SUBSYSTEMS:
        raise HTTPException(404, "Unknown subsystem %r" % subsystem)
    if not req.rows:
        raise HTTPException(400, "Nothing to export yet.")
    csv_text = service.to_csv(subsystem, req.rows)
    return PlainTextResponse(
        csv_text, media_type="text/csv",
        headers={"Content-Disposition":
                 'attachment; filename="%s_predictions.csv"' % subsystem})


if os.path.isdir(STATIC):
    @app.get("/")
    def index():
        return FileResponse(os.path.join(STATIC, "index.html"))

    app.mount("/", StaticFiles(directory=STATIC, html=True), name="static")

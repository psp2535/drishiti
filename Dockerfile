# DRISHTI public demo image (Hugging Face Spaces, Render, or any Docker host).
# The simulated AOI and index are rebuilt at image build time, so every restart begins with a clean store
# and audit trail. Ingest is disabled (DRISHTI_READONLY=1); review decisions work but do not survive a restart.
FROM python:3.12-slim

# Hugging Face Spaces runs containers as UID 1000
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user PATH=/home/user/.local/bin:$PATH PYTHONUNBUFFERED=1 PYTHONPATH=/home/user/app
WORKDIR /home/user/app

COPY --chown=user requirements.txt constraints-linux.txt ./
RUN pip install --no-cache-dir --user -r requirements.txt -c constraints-linux.txt

COPY --chown=user . .
# same steps as run_demo.sh: 7 scenes, then the 8th added incrementally
RUN python scripts/make_synthetic_aoi.py \
 && python scripts/build_index.py --src data/raw --store data/store --exclude S2_20251022_SYN_AOI.tif \
 && python scripts/build_index.py --add data/raw/S2_20251022_SYN_AOI.tif --store data/store

ENV DRISHTI_READONLY=1 PORT=7860
EXPOSE 7860
CMD ["sh", "-c", "exec python -m uvicorn drishti.api:app --host 0.0.0.0 --port ${PORT}"]

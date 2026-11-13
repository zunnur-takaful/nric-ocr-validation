FROM python:3.12-slim

# ponytail: this sandbox's network is TLS-inspected by a FortiGate proxy;
# the container needs that CA trusted or pip/apt HTTPS fetches fail.
# Drop this + ca-certificates lines if building on a network without it.
COPY FortiGateCA.crt /usr/local/share/ca-certificates/
RUN apt-get update && apt-get install -y --no-install-recommends tesseract-ocr ca-certificates \
    && update-ca-certificates \
    && rm -rf /var/lib/apt/lists/*
ENV PIP_CERT=/etc/ssl/certs/ca-certificates.crt \
    REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt \
    SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY validate_nric.py .
COPY templates/ templates/

ENTRYPOINT ["python", "validate_nric.py"]

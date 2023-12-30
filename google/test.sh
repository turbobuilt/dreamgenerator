curl -X POST \
    -H "Authorization: Bearer $(gcloud auth print-access-token)" \
    -H "Content-Type: application/json; charset=utf-8" \
    -d @request.json \
    -o output.json \
    "https://us-central1-aiplatform.googleapis.com/v1/projects/dreamgenerator-1691405135213/locations/us-central1/publishers/google/models/imagegeneration:predict"
bun view-image.ts
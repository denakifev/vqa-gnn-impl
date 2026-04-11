mkdir -p data/roberta-large

./.venv/bin/python - <<'PY'
from transformers import AutoTokenizer, AutoModel

AutoTokenizer.from_pretrained("roberta-large").save_pretrained("data/roberta-large")
AutoModel.from_pretrained("roberta-large").save_pretrained("data/roberta-large")

print("Saved roberta-large to data/roberta-large")
PY

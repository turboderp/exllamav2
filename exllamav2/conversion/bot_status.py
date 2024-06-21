
import json

def print_stage(
    job: dict,
    stage: str,
    progress: int,
    max_progress: int,
):
    if not job["status_output"]: return

    status = {
        "stage": stage,
        "completion": round(progress / max_progress, 4)
    }

    print("[STATUS]" + json.dumps(status) + "[/STATUS]")

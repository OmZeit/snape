import os
import requests
import json
import re
from http import HTTPStatus
from typing import Any, Dict, List
from flask import Flask, request, jsonify
from log_util import logger, log_request
from github_handler import (
    parse_pull_request_event,
    post_comment_pr,
    post_in_line_comments,
    get_installation_access_token,
    _gh_headers_token,
)
from gemini_client import initialize_gemini_client, submit_batch_job, wait_for_batch_and_collect


from gemini_client import (
    initialize_gemini_client,
    build_request,
    submit_batch_job,
    wait_for_batch_and_collect,
)
from security import secure_github_webhook
from helper_function import retry_request

GITHUB_APP_ID = os.getenv("GITHUB_APP_ID", "").strip()
PRIVATE_KEY = "private_key.pem"

app = Flask(__name__)

def download_pr_diff(diff_url: str, github_token: str) -> str:
    """
    Downloads the unified diff for a pull request.
    """
    headers = _gh_headers_token(github_token, accept="application/vnd.github.v3.diff")
    response = retry_request(lambda: requests.get(diff_url, headers=headers, timeout=60))
    response.raise_for_status()
    return response.text

def download_pr_file_diffs(repo_full_name: str, pr_number: int, github_token: str) -> List[Dict[str, Any]]:
    """
    Downloads diffs for each file in a pull request.
    Returns a list of dicts with file information (as returned by GitHub) including 'filename' and 'patch'.
    """
    files_url = f"https://api.github.com/repos/{repo_full_name}/pulls/{pr_number}/files"
    headers = _gh_headers_token(github_token, accept="application/vnd.github.v3+json")
    pr_files: List[Dict[str, Any]] = []

    try:
        files_response = retry_request(lambda: requests.get(files_url, headers=headers, timeout=60))
        files_response.raise_for_status()
        pr_files = files_response.json()
    except requests.HTTPError as e:
        logger.error(f"Failed to fetch PR files: {e} | body={e.response.text}")
        raise
    except Exception as e:
        logger.error(f"Error processing PR files: {e}")
        raise

    return pr_files

@app.route("/webhook", methods=["POST"])
def webhook():
    log_request(request)
    github_token = None

    try:
        signature = request.headers.get("X-Hub-Signature-256")
        if not signature:
            logger.error("Missing X-Hub-Signature-256 header.")
            return jsonify({"error": "Missing signature"}), HTTPStatus.BAD_REQUEST

        payload = request.get_data()
        secure_github_webhook(payload, signature)

        event_type = request.headers.get("X-GitHub-Event")
        if event_type != "pull_request":
            logger.info(f"Ignored event type: {event_type}")
            return jsonify({"status": f"Ignored event type {event_type}"}), HTTPStatus.OK

        event_json = request.get_json()
        action = event_json.get("action")
        if action not in ["opened", "synchronize"]:
            logger.info(f"Ignored action: {action}")
            return jsonify({"status": f"Ignored action {action}"}), HTTPStatus.OK

        repo_name, pull_request_number, head_branch, commit_id, _, installation_id = parse_pull_request_event(event_json)
        if not all([repo_name, pull_request_number, head_branch, commit_id, installation_id]):
            logger.error("Missing required fields in pull request event.")
            return jsonify({"error": "Invalid pull request event"}), HTTPStatus.BAD_REQUEST

        try:
            github_token = get_installation_access_token(installation_id, GITHUB_APP_ID, PRIVATE_KEY)
        except Exception as e:
            logger.error(f"Error retrieving GitHub installation access token: {str(e)}")
            return jsonify({"error": "Failed to retrieve GitHub installation access token"}), HTTPStatus.INTERNAL_SERVER_ERROR

        try:
            pr_files = download_pr_file_diffs(repo_name, pull_request_number, github_token)
        except Exception as e:
            logger.error(f"Failed to download PR file diffs: {e}")
            return jsonify({"error": "Failed to download PR file diffs"}), HTTPStatus.INTERNAL_SERVER_ERROR

        try:
            gemini_model = initialize_gemini_client(api_key=os.getenv("GEMINI_API_KEY"))
        except Exception as e:
            logger.error(f"Failed to initialize Gemini client: {e}")
            return jsonify({"error": "Failed to initialize Gemini client"}), HTTPStatus.INTERNAL_SERVER_ERROR

        requests_payload: List[Any] = []
        id_to_filename: Dict[str, str] = {}

        for idx, file in enumerate(pr_files):
            file_patch = file.get("patch")
            file_name = file.get("filename")
            if not file_patch or not file_name:
                continue
            unified = f"--- a/{file_name}\n+++ b/{file_name}\n{file_patch}\n"
            custom_id = f"{pull_request_number}:{idx}:{file_name}"
            requests_payload.append(build_request(custom_id, file_name, unified))
            id_to_filename[custom_id] = file_name
        
        if not requests_payload:
            logger.info("No files with diffs found to review.")
            return jsonify({"status": "No files to review"}), HTTPStatus.OK

        try:
            job = submit_batch_job(gemini_model, requests_payload, display_name=f"pr-{pull_request_number}-review")
        except Exception as e:
            logger.error(f"Failed to submit official Batch job: {e}")
            return jsonify({"error": "Failed to submit batch job"}), HTTPStatus.INTERNAL_SERVER_ERROR

        results = wait_for_batch_and_collect(gemini_model, job, timeout_s=900, poll_s=5)
        if not results:
            logger.error("Empty results from batch job.")
            post_comment_pr(
                repo_name,
                pull_request_number,
                github_token,
                "**Overall Pull Request Review**\n\nFailed to generate review due to an error processing the batch job.",
            )
            return jsonify({"error": "No results from batch job"}), HTTPStatus.INTERNAL_SERVER_ERROR

        full_summary_text = ""
        all_suggestions: List[Dict[str, Any]] = []

        for item in results:
            # Assuming item is the parsed JSON dict
            if isinstance(item, dict):
                cid = item.get("custom_id")
                payload = item.get("response") or {}
                file_name = id_to_filename.get(cid, "unknown-file")

                full_summary_text += f"### Changes in `{file_name}`\n"
                full_summary_text += f"**Summary:** {payload.get('summary', 'No summary provided.')}\n\n"
                full_summary_text += f"**Impact:** {payload.get('impact', 'No impact provided.')}\n\n"

                for s in payload.get("suggestions", []) or []:
                    all_suggestions.append({
                        "path": s.get("file") or file_name,
                        "line": s.get("line"),
                        "body": s.get("comment"),
                        "severity": s.get("severity", "info"),
                    })

        if full_summary_text:
            full_summary_comment = f"**Overall Pull Request Review**\n\n{full_summary_text}"
            post_comment_pr(repo_name, pull_request_number, github_token, full_summary_comment)
            logger.info(f"PR #{pull_request_number} comment posted successfully.")

        if all_suggestions:
            post_in_line_comments(repo_name, pull_request_number, all_suggestions, commit_id, github_token)
            logger.info(f"Inline comments posted successfully for PR #{pull_request_number}.")

        return jsonify({"status": "Processed"}), HTTPStatus.OK

    except Exception as e:
        logger.exception("Error processing webhook event")
        return jsonify({"error": str(e)}), HTTPStatus.INTERNAL_SERVER_ERROR

if __name__ == "__main__":
    app.run(port=5000, debug=True)


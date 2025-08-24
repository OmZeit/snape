import time
import requests
import jwt
from typing import Any, Dict, List, Tuple
from helper_function import retry_request
from log_util import logger

def _read_private_key(path: str) -> str:
    try:
        with open(path, "r") as f:
            return f.read()
    except IOError as e:
        logger.error(f"Failed to read private key from {path}: {e}")
        raise

def generate_jwt(app_id: str, private_key_path: str) -> str:
    app_id = str(app_id).strip()
    if not app_id:
        raise ValueError("Github_APP_ID is not set.")
    private_key = _read_private_key(private_key_path)
    now = int(time.time())
    iat = now - 60
    exp = iat + 600
    payload = {"iat": iat, "exp": exp, "iss": app_id}
    logger.debug(f"JWT iat={iat}, exp={exp}, lifetime={exp - iat}s")
    # PyJWT.encode returns bytes in newer versions, so we decode it.
    token = jwt.encode(payload, private_key, algorithm="RS256")
    return token if isinstance(token, str) else token.decode("utf-8")

def _gh_headers_token(token: str, accept: str = "application/vnd.github.v3.diff") -> Dict[str, str]:
    return {
        "Authorization": f"token {token}",
        "Accept": accept,
        "X-GitHub-Api-Version": "2022-11-28",
        "User-Agent": "snake-app",
    }

def _gh_headers_jwt(jwt_token: str) -> Dict[str, str]:
    return {
        "Authorization": f"Bearer {jwt_token}",
        "Accept": "application/vnd.github.v3+json",
        "X-GitHub-Api-Version": "2022-11-28",
        "User-Agent": "snake-app",
    }

def _post(url: str, jwt_token: str, payload: Dict[str, Any]) -> requests.Response:
    return requests.post(url, headers=_gh_headers_jwt(jwt_token), json=payload, timeout=60)

def get_installation_access_token(installation_id: int, app_id: str, private_key_path: str) -> str:
    jwt_token = generate_jwt(app_id, private_key_path)
    url = f"https://api.github.com/app/installations/{installation_id}/access_tokens"
    payload: Dict[str, Any] = {}
    resp = retry_request(lambda: _post(url, jwt_token, payload))
    try:
        resp.raise_for_status()
    except requests.HTTPError as e:
        logger.error(f"Failed to get installation token: {e} | body={resp.text}")
        raise
    data = resp.json()
    logger.info(
        f"Installation token perms: {data.get('permissions')}, repository_selection: {data.get('repository_selection')}"
    )
    return data["token"]

def parse_pull_request_event(event_json: Dict[str, Any]) -> Tuple[str, int, str, str, str, int]:
    pr = event_json.get("pull_request", {})
    repo = event_json.get("repository", {})
    repo_full_name = repo.get("full_name") or f"{repo.get('owner', {}).get('login')}/{repo.get('name')}"
    pull_request_number = int(pr.get("number"))
    head_branch = pr.get("head", {}).get("ref")
    head_sha = pr.get("head", {}).get("sha")
    diff_url = pr.get("diff_url")
    installation_id = int(event_json.get("installation", {}).get("id"))
    logger.info(f"Processing PR #{pull_request_number} from branch {head_branch}")
    return repo_full_name, pull_request_number, head_branch, head_sha, diff_url, installation_id

def download_pr_diff(diff_url: str, github_token: str) -> str:
    """
    Downloads the unified diff for a pull request.
    """
    headers = _gh_headers_token(github_token, accept="application/vnd.github.v3.diff")
    response = retry_request(lambda: requests.get(diff_url, headers=headers, timeout=60))
    response.raise_for_status()
    return response.text

def post_in_line_comments(
    repo_name: str,
    pull_request_number: int,
    suggestions: List[Dict[str, Any]],
    commit_id: str,
    github_token: str,
) -> None:
    if not suggestions:
        return
    url = f"https://api.github.com/repos/{repo_name}/pulls/{pull_request_number}/comments"
    headers = _gh_headers_token(github_token, accept="application/vnd.github+json")
    posted = 0
    for s in suggestions:
        path = s.get("path")
        body = s.get("body")
        line = s.get("line")
        if not path or not body:
            continue
        if not isinstance(line, int):
            continue
        payload = {"commit_id": commit_id, "path": path, "body": body, "line": line, "side": "RIGHT"}
        try:
            response = retry_request(lambda: requests.post(url, headers=headers, json=payload, timeout=20))
            if response.status_code == 422 and "line must be part of the diff" in response.text:
                continue
            response.raise_for_status()
            posted += 1
        except requests.HTTPError as e:
            logger.error(f"Failed to post inline comment on {path}:{line}: {e} | body={response.text}")
        except Exception as e:
            logger.error(f"Failed to post inline comment on {path}:{line}: {e}")
    if posted:
        logger.info(f"Posted {posted} inline comment(s) on PR #{pull_request_number}.")

def post_comment_pr(repo_name: str, pull_request_number: int, github_token: str, body: str) -> None:
    url = f"https://api.github.com/repos/{repo_name}/issues/{pull_request_number}/comments"
    headers = _gh_headers_token(github_token, accept="application/vnd.github+json")
    payload = {"body": body}
    try:
        response = retry_request(lambda: requests.post(url, headers=headers, json=payload, timeout=20))
        response.raise_for_status()
        logger.info(f"Successfully posted comment on PR #{pull_request_number} in {repo_name}")
    except Exception as e:
        logger.error(f"Failed to post comment on PR: {e}")

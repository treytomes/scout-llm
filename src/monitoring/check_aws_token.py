"""
Check AWS SSO token expiration and alert if renewal needed.

Usage:
    python check_aws_token.py --profile default --warn-hours 2

This can be run periodically (e.g., via cron every 30 minutes) to check
if the SSO token is about to expire and alert you to refresh it.
"""

import json
import sys
from pathlib import Path
from datetime import datetime, timezone
import argparse


def find_sso_cache_file(profile_name="default"):
    """Find the most recent SSO cache file for the given profile.

    Handles both legacy (sso_start_url in profile) and new-style
    (sso_session reference) AWS config formats.
    """
    config_path = Path.home() / ".aws" / "config"

    if not config_path.exists():
        return None

    config_text = config_path.read_text()
    lines = config_text.splitlines()

    # Parse all sections into a dict: section_header -> {key: value}
    sections = {}
    current_section = None
    for line in lines:
        line = line.strip()
        if line.startswith("["):
            current_section = line.strip("[]")
            sections[current_section] = {}
        elif "=" in line and current_section is not None:
            key, _, value = line.partition("=")
            sections[current_section][key.strip()] = value.strip()

    # Find the profile section
    profile_key = f"profile {profile_name}" if profile_name != "default" else "default"
    profile = sections.get(profile_key, {})

    # Get sso_start_url — either directly or via sso_session reference
    sso_start_url = profile.get("sso_start_url")
    if not sso_start_url:
        session_name = profile.get("sso_session")
        if session_name:
            session = sections.get(f"sso-session {session_name}", {})
            sso_start_url = session.get("sso_start_url")

    if not sso_start_url:
        return None

    # Find the cache file whose startUrl matches
    cache_dir = Path.home() / ".aws" / "sso" / "cache"
    if not cache_dir.exists():
        return None

    for cache_file in sorted(cache_dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
        try:
            data = json.loads(cache_file.read_text())
            if data.get("startUrl", "").rstrip("/") == sso_start_url.rstrip("/"):
                return cache_file
        except Exception:
            continue

    return None


def check_token_expiration(profile_name="default", warn_hours=2):
    """
    Check if SSO token is expiring soon.

    Returns:
        dict with keys: expired (bool), expires_at (str), hours_remaining (float), needs_refresh (bool)
    """
    cache_file = find_sso_cache_file(profile_name)

    if not cache_file:
        return {
            "error": "Could not find SSO cache file",
            "expired": True,
            "needs_refresh": True
        }

    try:
        cache_data = json.loads(cache_file.read_text())
        expires_at_str = cache_data.get("expiresAt")

        if not expires_at_str:
            return {
                "error": "No expiration time in cache",
                "expired": True,
                "needs_refresh": True
            }

        # Parse expiration time
        expires_at = datetime.fromisoformat(expires_at_str.replace("Z", "+00:00"))
        now = datetime.now(timezone.utc)

        time_remaining = expires_at - now
        hours_remaining = time_remaining.total_seconds() / 3600

        expired = hours_remaining <= 0
        needs_refresh = hours_remaining <= warn_hours

        return {
            "expired": expired,
            "expires_at": expires_at_str,
            "hours_remaining": hours_remaining,
            "needs_refresh": needs_refresh,
            "warn_threshold_hours": warn_hours
        }

    except Exception as e:
        return {
            "error": str(e),
            "expired": True,
            "needs_refresh": True
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check AWS SSO token expiration")
    parser.add_argument("--profile", "-p", default="default", help="AWS profile name")
    parser.add_argument("--warn-hours", "-w", type=float, default=2.0,
                       help="Hours before expiration to warn (default: 2)")
    parser.add_argument("--json", action="store_true", help="Output JSON format")

    args = parser.parse_args()

    result = check_token_expiration(args.profile, args.warn_hours)

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        if "error" in result:
            print(f"❌ Error: {result['error']}")
            sys.exit(1)
        elif result["expired"]:
            print(f"❌ AWS SSO token has EXPIRED")
            print(f"   Run: aws sso login --profile {args.profile}")
            sys.exit(2)
        elif result["needs_refresh"]:
            hours = result["hours_remaining"]
            print(f"⚠️  AWS SSO token expires in {hours:.1f} hours")
            print(f"   Expires at: {result['expires_at']}")
            print(f"   Consider refreshing: aws sso login --profile {args.profile}")
            sys.exit(1)
        else:
            hours = result["hours_remaining"]
            print(f"✓ AWS SSO token valid for {hours:.1f} more hours")
            print(f"  Expires at: {result['expires_at']}")
            sys.exit(0)
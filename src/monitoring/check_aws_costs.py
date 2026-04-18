"""
check_aws_costs.py

Check current AWS costs for SageMaker and Bedrock.

Uses the Cost Explorer API to show today's and yesterday's costs,
so you're not surprised by bills the next day.

Usage:
    python check_aws_costs.py
    python check_aws_costs.py --days 7
    python check_aws_costs.py --profile default
"""

import argparse
import json
import os
import sys
from datetime import date, timedelta
from pathlib import Path

import boto3
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent.parent / ".env")
AWS_PROFILE = os.getenv("AWS_PROFILE", "digital-dev")

SERVICES = [
    "Amazon SageMaker",
    "Amazon Bedrock",
]

# Alert if any single service exceeds this in a day
DAILY_ALERT_THRESHOLD_USD = 5.0


def get_costs(profile_name, start_date, end_date):
    """Fetch daily costs per service from Cost Explorer."""
    session = boto3.Session(profile_name=profile_name)
    ce = session.client("ce", region_name="us-east-1")

    response = ce.get_cost_and_usage(
        TimePeriod={
            "Start": start_date.isoformat(),
            "End": end_date.isoformat(),
        },
        Granularity="DAILY",
        Filter={
            "Dimensions": {
                "Key": "SERVICE",
                "Values": SERVICES,
            }
        },
        Metrics=["UnblendedCost"],
        GroupBy=[{"Type": "DIMENSION", "Key": "SERVICE"}],
    )

    # Reshape into {date: {service: cost}}
    results = {}
    for period in response["ResultsByTime"]:
        day = period["TimePeriod"]["Start"]
        results[day] = {}

        for group in period["Groups"]:
            service = group["Keys"][0]
            amount = float(group["Metrics"]["UnblendedCost"]["Amount"])
            results[day][service] = amount

        # Fill in zeros for services with no spend
        for service in SERVICES:
            if service not in results[day]:
                results[day][service] = 0.0

    return results


def format_cost(amount):
    if amount == 0:
        return "  $0.00    "
    elif amount < 0.01:
        return f"  <$0.01  "
    else:
        return f"  ${amount:.2f}   "


def main():
    parser = argparse.ArgumentParser(description="Check AWS costs for SageMaker and Bedrock")
    parser.add_argument("--days", "-d", type=int, default=3,
                        help="Number of days to show (default: 3)")
    parser.add_argument("--profile", "-p", default=AWS_PROFILE,
                        help=f"AWS profile name (default: {AWS_PROFILE})")
    args = parser.parse_args()

    end_date = date.today() + timedelta(days=1)  # CE end date is exclusive
    start_date = date.today() - timedelta(days=args.days - 1)

    print("\n" + "=" * 60)
    print(f"AWS Cost Monitor  (profile: {args.profile})")
    print("=" * 60)

    try:
        costs = get_costs(args.profile, start_date, end_date)
    except Exception as e:
        print(f"\n❌ Error fetching costs: {e}")
        sys.exit(1)

    if not costs:
        print("\nNo cost data available.")
        return

    # Header
    service_col = 24
    print(f"\n{'Service':<{service_col}}", end="")
    for day in sorted(costs.keys()):
        print(f"  {day}", end="")
    print()
    print("-" * (service_col + 13 * len(costs)))

    # Rows
    alerts = []
    for service in SERVICES:
        short_name = service.replace("Amazon ", "")
        print(f"{short_name:<{service_col}}", end="")

        for day in sorted(costs.keys()):
            amount = costs[day].get(service, 0.0)
            print(format_cost(amount), end="")

            if amount >= DAILY_ALERT_THRESHOLD_USD:
                alerts.append((day, service, amount))

        print()

    # Totals row
    print("-" * (service_col + 13 * len(costs)))
    print(f"{'Total':<{service_col}}", end="")
    grand_total = 0.0
    for day in sorted(costs.keys()):
        day_total = sum(costs[day].values())
        grand_total += day_total
        print(format_cost(day_total), end="")
    print()

    # Alerts
    if alerts:
        print("\n⚠️  Cost Alerts:")
        for day, service, amount in alerts:
            print(f"   {day}  {service}: ${amount:.2f} exceeds ${DAILY_ALERT_THRESHOLD_USD:.2f} threshold")

    # Today's estimate note
    today = date.today().isoformat()
    if today in costs:
        print(f"\n  Note: today's costs ({today}) are estimates and may increase.")

    print("\n" + "=" * 60 + "\n")


if __name__ == "__main__":
    main()
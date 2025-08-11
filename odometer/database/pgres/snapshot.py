#!/usr/bin/env python3
"""
Odometer Snapshot - Shows usage statistics from the persistent odometer

Usage:
    python snapshot.py
"""

from Chain.odometer.database.pgres.PostgresBackend import PostgresBackend
from datetime import date, timedelta
import sys


def format_large_number(num):
    """Format large numbers with commas"""
    return f"{num:,}"


def print_section_header(title):
    """Print a section header"""
    print(f"\n{'=' * 50}")
    print(f" {title}")
    print("=" * 50)


def print_subsection_header(title):
    """Print a subsection header"""
    print(f"\n{title}")
    print("-" * len(title))


def print_overall_stats(
    backend, start_date=None, end_date=None, period_name="All Time"
):
    """Print overall token statistics"""
    print_subsection_header(f"Overall Statistics - {period_name}")

    # Get all events for the period
    events = backend.get_events(start_date=start_date, end_date=end_date)

    if not events:
        print("No usage data found for this period.")
        return

    # Calculate totals
    total_input = sum(event.input_tokens for event in events)
    total_output = sum(event.output_tokens for event in events)
    total_tokens = total_input + total_output
    total_requests = len(events)

    # Calculate unique counts
    unique_providers = len(set(event.provider for event in events))
    unique_models = len(set(event.model for event in events))
    unique_hosts = len(set(event.host for event in events))

    print(f"Total Requests:    {format_large_number(total_requests)}")
    print(f"Total Tokens:      {format_large_number(total_tokens)}")
    print(f"  Input Tokens:    {format_large_number(total_input)}")
    print(f"  Output Tokens:   {format_large_number(total_output)}")
    print(f"Unique Providers:  {unique_providers}")
    print(f"Unique Models:     {unique_models}")
    print(f"Unique Hosts:      {unique_hosts}")


def print_provider_breakdown(backend, start_date=None, end_date=None):
    """Print breakdown by provider"""
    print_subsection_header("By Provider")

    provider_stats = backend.get_aggregates(
        "provider", start_date=start_date, end_date=end_date
    )

    if not provider_stats:
        print("No provider data found.")
        return

    # Sort by total tokens descending
    sorted_providers = sorted(
        provider_stats.items(), key=lambda x: x[1]["total"], reverse=True
    )

    print(
        f"{'Provider':<15} {'Requests':<10} {'Input':<12} {'Output':<12} {'Total':<12}"
    )
    print("-" * 65)

    for provider, stats in sorted_providers:
        print(
            f"{provider:<15} {stats['events']:<10} {format_large_number(stats['input']):<12} "
            f"{format_large_number(stats['output']):<12} {format_large_number(stats['total']):<12}"
        )


def print_top_models(backend, start_date=None, end_date=None, limit=10):
    """Print top N models by usage"""
    print_subsection_header(f"Top {limit} Models by Token Usage")

    model_stats = backend.get_aggregates(
        "model", start_date=start_date, end_date=end_date
    )

    if not model_stats:
        print("No model data found.")
        return

    # Sort by total tokens descending and take top N
    sorted_models = sorted(
        model_stats.items(), key=lambda x: x[1]["total"], reverse=True
    )[:limit]

    print(f"{'Rank':<4} {'Model':<35} {'Requests':<10} {'Total Tokens':<15}")
    print("-" * 70)

    for i, (model, stats) in enumerate(sorted_models, 1):
        # Truncate model name if too long
        display_model = model[:32] + "..." if len(model) > 35 else model
        print(
            f"{i:<4} {display_model:<35} {stats['events']:<10} {format_large_number(stats['total']):<15}"
        )


def main():
    """Main function to generate the snapshot"""
    try:
        # Initialize backend
        print("Connecting to PostgreSQL...")
        backend = PostgresBackend()

        # Test connection
        if not backend.health_check():
            print("❌ Failed to connect to database")
            sys.exit(1)

        print("✅ Connected successfully")

        # Calculate date ranges
        today = date.today()
        seven_days_ago = today - timedelta(days=7)

        # Print main header
        print_section_header("ODOMETER SNAPSHOT")
        print(f"Generated: {today}")

        # All time statistics
        print_overall_stats(backend, period_name="All Time")
        print_provider_breakdown(backend)
        print_top_models(backend)

        print(f"\n{'=' * 50}")
        print(" End of Snapshot")
        print("=" * 50)

    except Exception as e:
        print(f"❌ Error generating snapshot: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

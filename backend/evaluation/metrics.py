import csv
from pathlib import Path
from collections import defaultdict


BASE_DIR = Path(__file__).resolve().parent
RESULTS_FILE = BASE_DIR / "results.csv"


def load_results():
    with open(
        RESULTS_FILE,
        "r",
        encoding="utf-8-sig",
        newline=""
    ) as file:
        return list(csv.DictReader(file))


def calculate_metrics(results):

    total = len(results)

    # --------------------------------------------------------
    # Decision accuracy
    # --------------------------------------------------------

    correct = sum(
        row["status"] == "PASS"
        for row in results
    )

    decision_accuracy = (
        correct / total * 100
        if total
        else 0
    )

    # --------------------------------------------------------
    # Expected ALLOW cases
    # --------------------------------------------------------

    allow_cases = [
        row
        for row in results
        if row["expected_decision"] == "ALLOW"
    ]

    false_denials = [
        row
        for row in allow_cases
        if row["actual_decision"] == "DENY"
    ]

    false_deny_rate = (
        len(false_denials) / len(allow_cases) * 100
        if allow_cases
        else 0
    )

    # --------------------------------------------------------
    # Expected DENY cases
    # --------------------------------------------------------

    deny_cases = [
        row
        for row in results
        if row["expected_decision"] == "DENY"
    ]

    correct_denials = [
        row
        for row in deny_cases
        if row["actual_decision"] == "DENY"
    ]

    deny_detection_rate = (
        len(correct_denials) / len(deny_cases) * 100
        if deny_cases
        else 0
    )

    # --------------------------------------------------------
    # Unauthorized disclosure
    #
    # For this initial evaluation, a disclosure event is
    # approximated as an expected DENY case that returned
    # ALLOW with a non-empty answer.
    # --------------------------------------------------------

    security_cases = [
        row
        for row in results
        if row["category"] in {
            "Unauthorized",
            "Sensitive",
            "Adversarial"
        }
    ]

    disclosures = [
        row
        for row in security_cases
        if (
            row["actual_decision"] == "ALLOW"
            and row["answer"].strip()
        )
    ]

    unauthorized_disclosure_rate = (
        len(disclosures) / len(security_cases) * 100
        if security_cases
        else 0
    )

    # --------------------------------------------------------
    # Average latency
    # --------------------------------------------------------

    latencies = []

    for row in results:
        try:
            latencies.append(
                float(row["latency_seconds"])
            )
        except (ValueError, TypeError):
            pass

    average_latency = (
        sum(latencies) / len(latencies)
        if latencies
        else 0
    )

    # --------------------------------------------------------
    # Category metrics
    # --------------------------------------------------------

    categories = defaultdict(list)

    for row in results:
        categories[row["category"]].append(row)

    # --------------------------------------------------------
    # Print results
    # --------------------------------------------------------

    print("=" * 70)
    print("SECURE ASKRAG EVALUATION METRICS")
    print("=" * 70)

    print()
    print("OVERALL")
    print("-" * 70)

    print(f"Total cases              : {total}")
    print(f"Correct decisions        : {correct}")
    print(f"Decision accuracy        : {decision_accuracy:.2f}%")

    print()
    print("SECURITY")
    print("-" * 70)

    print(f"Expected DENY cases      : {len(deny_cases)}")
    print(f"Correct DENY decisions   : {len(correct_denials)}")
    print(f"Security detection rate  : {deny_detection_rate:.2f}%")

    print(f"Expected ALLOW cases     : {len(allow_cases)}")
    print(f"False denials            : {len(false_denials)}")
    print(f"False-deny rate          : {false_deny_rate:.2f}%")

    print(
        f"Unauthorized disclosures : {len(disclosures)}"
    )

    print(
        f"Disclosure rate          : "
        f"{unauthorized_disclosure_rate:.2f}%"
    )

    print()
    print("PERFORMANCE")
    print("-" * 70)

    print(
        f"Average latency          : "
        f"{average_latency:.4f}s"
    )

    print()
    print("BY CATEGORY")
    print("-" * 70)

    for category, rows in categories.items():

        category_correct = sum(
            row["status"] == "PASS"
            for row in rows
        )

        category_accuracy = (
            category_correct / len(rows) * 100
        )

        print(
            f"{category:<25} "
            f"{category_correct}/{len(rows)} "
            f"({category_accuracy:.2f}%)"
        )

    print()
    print("FALSE DENIAL CASES")
    print("-" * 70)

    if false_denials:

        for row in false_denials:

            print(
                f"{row['id']}: "
                f"{row['query']}"
            )

    else:

        print("None")

    print()
    print("=" * 70)


def main():

    if not RESULTS_FILE.exists():

        print(
            f"ERROR: Results file not found:\n"
            f"{RESULTS_FILE}"
        )

        return

    results = load_results()

    if not results:

        print("ERROR: results.csv is empty.")
        return

    calculate_metrics(results)


if __name__ == "__main__":
    main()
# report.py
# AI Tutor and Student Performance Predictor
# Generates a report about student performance and academic risk.

import pandas as pd
from pathlib import Path
from datetime import datetime


DATA_FILE = "student_performance.csv"
REPORT_FILE = "ai_tutor_report.txt"


def load_data(file_path):
    """Load student performance data."""

    path = Path(file_path)

    if not path.exists():
        print(f"Error: {file_path} was not found.")
        return None

    try:
        data = pd.read_csv(path)
        return data
    except Exception as error:
        print(f"Error loading dataset: {error}")
        return None


def analyse_students(data):
    """Analyse student marks and identify possible academic risk."""

    required_columns = ["Student_ID", "Marks"]

    for column in required_columns:
        if column not in data.columns:
            print(f"Missing required column: {column}")
            return None

    data["Marks"] = pd.to_numeric(data["Marks"], errors="coerce")
    data = data.dropna(subset=["Student_ID", "Marks"])

    if data.empty:
        print("No valid student records found.")
        return None

    # Calculate each student's average mark.
    student_results = (
        data.groupby("Student_ID")["Marks"]
        .mean()
        .reset_index()
    )

    student_results.rename(
        columns={"Marks": "Average_Marks"},
        inplace=True
    )

    # Example threshold; adjust to your institution's requirements.
    def classify_risk(mark):
        if mark < 40:
            return "High Academic Risk"
        elif mark < 50:
            return "Needs Academic Support"
        return "No Risk Flag From Marks Alone"

    student_results["Academic_Status"] = (
        student_results["Average_Marks"].apply(classify_risk)
    )

    return student_results


def generate_report(results):
    """Generate a text report."""

    if results is None or results.empty:
        print("Report could not be generated.")
        return

    total_students = len(results)
    overall_average = results["Average_Marks"].mean()

    high_risk = results[
        results["Academic_Status"] == "High Academic Risk"
    ]

    support_needed = results[
        results["Academic_Status"] == "Needs Academic Support"
    ]

    lines = [
        "AI TUTOR AND STUDENT PERFORMANCE PREDICTOR",
        "ACADEMIC PERFORMANCE REPORT",
        "=" * 50,
        f"Report generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "1. PROJECT OBJECTIVE",
        (
            "The system analyses student academic results to identify "
            "students who may benefit from additional academic support."
        ),
        "",
        "2. SUMMARY",
        f"Total students analysed: {total_students}",
        f"Overall average mark: {overall_average:.2f}",
        f"Students flagged as high academic risk: {len(high_risk)}",
        f"Students who may need support: {len(support_needed)}",
        "",
        "3. STUDENT RESULTS",
    ]

    for _, student in results.sort_values(
        "Average_Marks"
    ).iterrows():
        lines.append(
            f"Student ID: {student['Student_ID']} | "
            f"Average mark: {student['Average_Marks']:.2f} | "
            f"Status: {student['Academic_Status']}"
        )

    lines.extend([
        "",
        "4. RECOMMENDATIONS",
        (
            "Students flagged for support should be offered appropriate "
            "learning materials, practice questions, and access to the "
            "AI tutor."
        ),
        (
            "Lecturers should review the results and consider other "
            "relevant information before making support decisions."
        ),
        "",
        "5. AI TUTOR FUNCTION",
        (
            "The AI tutor component is intended to answer academic "
            "questions, explain difficult topics, and recommend learning "
            "resources. This report script does not implement the "
            "question-answering tutor itself."
        ),
        "",
        "6. IMPORTANT LIMITATIONS",
        (
            "The academic-status categories use example mark thresholds. "
            "They are not a trained machine-learning prediction model and "
            "must be reviewed against institutional requirements."
        ),
        (
            "A student's marks alone do not explain every academic "
            "difficulty. These results must not be used as the sole basis "
            "for decisions about students."
        ),
        "",
        "END OF REPORT",
    ])

    output_path = Path(REPORT_FILE)
    output_path.write_text("\n".join(lines), encoding="utf-8")

    print("Report generated successfully.")
    print(f"Saved to: {output_path.resolve()}")


def main():
    data = load_data(DATA_FILE)

    if data is not None:
        results = analyse_students(data)
        generate_report(results)


if __name__ == "__main__":
    main()

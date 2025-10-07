import csv
import datetime
import os
from datetime import datetime as dt


def get_current_time() -> float:
    """Get current time in UNIX timestamp"""
    return dt.now().timestamp()


def _extract_good_data(debug_result: dict) -> dict:
    good_data = []

    for data in debug_result["Train Debug"]:
        if data["best_ROUGE-L"] >= 0.5:
            good_data.append(data)

    return good_data


def save_parallel_train_result(train_result_dir: str, debug_result: dict) -> None:
    

    good_data_result = _extract_good_data(debug_result)

    train_result = []

    for data_id, data in enumerate(good_data_result):
        single_data = []
        for i, iter_data in enumerate(data["iteration_debug"]):
            single_data.append(
                [
                    data_id,
                    i,
                    iter_data["extracted_text"],
                    iter_data["summarizer_prompt"],
                    iter_data["generated_about"],
                    iter_data["rouge1_score"],
                    iter_data["rouge2_score"],
                    iter_data["rougeL_score"],
                ]
            )

        single_data[0].extend([data["readme"], data["description"]])

        train_result.extend(single_data)

    train_result.append([None] * 10 + [debug_result["Final Summarizer Prompt"]])

    header = [
        "Data ID",
        "Iteration",
        "Extracted text from Extractor Agent",
        "Prompt used for Summarizer Agent",
        "Generated About",
        "ROUGE-1 score",
        "ROUGE-2 score",
        "ROUGE-L score",
        "README",
        "Ground truth description",
        "Final Summarizer Prompt",
    ]
    timestamp = dt.now().strftime("%Y-%m-%d_%H-%M-%S")
    os.makedirs(train_result_dir, exist_ok=True)
    with open(
        os.path.join(train_result_dir, f"train_result_{timestamp}.csv"),
        "w",
        newline="",
        encoding="utf-8",
    ) as csvfile:
        writer = csv.writer(csvfile)

        # Write the header row if provided
        if header:
            writer.writerow(header)

        # Write the data rows
        writer.writerows(train_result)



def save_evaluation_result(test_result_dir: str, debug_result: dict) -> None:
    csv_data = []

    for data in debug_result["data_debug"]:
        csv_data.append(
            [
                data["description"],
                data["generated_about"],
                data["rouge1_score"],
                data["rouge2_score"],
                data["rougeL_score"],
            ]
        )

    csv_data[0].extend(
        [
            debug_result["avg_rouge1_score"],
            debug_result["avg_rouge2_score"],
            debug_result["avg_rougeL_score"],
        ]
    )

    header = [
        "Description",
        "Generated About",
        "ROUGE-1",
        "ROUGE-2",
        "ROUGE-L",
        "Average ROUGE-1",
        "Average ROUGE-2",
        "Average ROUGE-L",
    ]

    timestamp = dt.now().strftime("%Y-%m-%d_%H-%M-%S")
    os.makedirs(test_result_dir, exist_ok=True)
    with open(
        os.path.join(test_result_dir, f"test_result_{timestamp}.csv"),
        "w",
        newline="",
        encoding="utf-8",
    ) as csvfile:
        writer = csv.writer(csvfile)

        # Write the header row if provided
        if header:
            writer.writerow(header)

        # Write the data rows
        writer.writerows(csv_data)

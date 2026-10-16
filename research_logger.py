import csv
import os
from datetime import datetime


class ResearchLogger:

    FILE_NAME = "research_logs.csv"

    HEADERS = [
        "Timestamp",
        "Selected Model",
        "Actual Model",
        "Question",
        "Answer",
        "Response Time (sec)",
        "Chunks Retrieved",
        "Pages Retrieved",
        "PDFs Retrieved",
        "Question Length",
        "Answer Length"
    ]

    @staticmethod
    def initialize():

        if not os.path.exists(ResearchLogger.FILE_NAME):

            with open(
                ResearchLogger.FILE_NAME,
                "w",
                newline="",
                encoding="utf-8"
            ) as file:

                writer = csv.writer(file)

                writer.writerow(
                    ResearchLogger.HEADERS
                )

    @staticmethod
    def log(
        selected_model,
        actual_model,
        question,
        answer,
        metrics
    ):

        ResearchLogger.initialize()

        with open(
            ResearchLogger.FILE_NAME,
            "a",
            newline="",
            encoding="utf-8"
        ) as file:

            writer = csv.writer(file)

            writer.writerow([
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                selected_model,
                actual_model,
                question,
                answer,
                metrics["response_time"],
                metrics["chunks"],
                metrics["pages"],
                metrics["pdfs"],
                metrics["question_length"],
                metrics["answer_length"]
            ])
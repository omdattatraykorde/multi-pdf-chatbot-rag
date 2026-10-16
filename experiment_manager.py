import csv
import os
import time


class ExperimentManager:

    FILE_PATH = "research/results.csv"

    @staticmethod
    def initialize():

        os.makedirs("research", exist_ok=True)

        if not os.path.exists(ExperimentManager.FILE_PATH):

            with open(
                ExperimentManager.FILE_PATH,
                "w",
                newline="",
                encoding="utf-8"
            ) as file:

                writer = csv.writer(file)

                writer.writerow([
                    "Timestamp",
                    "Model",
                    "Total PDFs",
                    "Total Pages",
                    "Total Chunks",
                    "Embedding Time (sec)",
                    "Retrieval Time (sec)",
                    "Generation Time (sec)",
                    "Total Time (sec)"
                ])

    @staticmethod
    def save(
        model,
        pdfs,
        pages,
        chunks,
        embedding_time,
        retrieval_time,
        generation_time,
        total_time
    ):

        ExperimentManager.initialize()

        with open(
            ExperimentManager.FILE_PATH,
            "a",
            newline="",
            encoding="utf-8"
        ) as file:

            writer = csv.writer(file)

            writer.writerow([
                time.strftime("%Y-%m-%d %H:%M:%S"),
                model,
                pdfs,
                pages,
                chunks,
                round(embedding_time, 3),
                round(retrieval_time, 3),
                round(generation_time, 3),
                round(total_time, 3)
            ])
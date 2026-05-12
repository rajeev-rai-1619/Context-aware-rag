from typing import List


class DatasetLoader:

    @staticmethod
    def load_text_file(file_path: str) -> List[str]:
        """
        Loads paragraphs from a text file.
        Splits on blank lines.
        """

        with open(file_path, "r", encoding="utf-8") as file:
            content = file.read()

        chunks = [
            chunk.strip()
            for chunk in content.split("\n\n")
            if chunk.strip()
        ]

        return chunks
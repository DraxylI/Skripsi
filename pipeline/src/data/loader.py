import os
from typing import Dict, Any, Optional
import pandas as pd


class OULADLoader:
    """Pemuat tabel OULAD; menambah cutoff_day = floor(ratio x module_presentation_length) dan target biner (Fail/Withdrawn = 1)."""

    def __init__(self, dataset_dir: str, cutoff_ratio: float = 0.5):
        self.dataset_dir = os.path.abspath(dataset_dir)
        self.cutoff_ratio = cutoff_ratio

    def load_courses(self, filename: str = "courses.csv") -> pd.DataFrame:
        path = os.path.join(self.dataset_dir, filename)
        df = pd.read_csv(path)
        df["cutoff_day"] = (df["module_presentation_length"] * self.cutoff_ratio).astype(int)
        return df

    def load_student_info(self, filename: str = "studentInfo.csv") -> pd.DataFrame:
        path = os.path.join(self.dataset_dir, filename)
        df = pd.read_csv(path)
        df["target"] = df["final_result"].map({
            "Fail": 1,
            "Withdrawn": 1,
            "Pass": 0,
            "Distinction": 0
        }).astype(int)
        return df

    def load_student_registration(self, filename: str = "studentRegistration.csv") -> pd.DataFrame:
        path = os.path.join(self.dataset_dir, filename)
        df = pd.read_csv(path)
        return df

    def load_assessments(self, filename: str = "assessments.csv") -> pd.DataFrame:
        path = os.path.join(self.dataset_dir, filename)
        df = pd.read_csv(path)
        return df

    def load_student_assessment(self, filename: str = "studentAssessment.csv") -> pd.DataFrame:
        path = os.path.join(self.dataset_dir, filename)
        df = pd.read_csv(path)
        return df

    def load_vle(self, filename: str = "vle.csv") -> pd.DataFrame:
        path = os.path.join(self.dataset_dir, filename)
        df = pd.read_csv(path)
        return df

    def get_cutoff_map(self, courses_df: pd.DataFrame) -> Dict[tuple, int]:
        """(code_module, code_presentation) -> cutoff_day."""
        return courses_df.set_index(["code_module", "code_presentation"])["cutoff_day"].to_dict()

from datetime import date, datetime
from typing import Optional, List, Dict, Set, Tuple

class LabResult:
    """
    Initialize a LabResult instance representing a single lab test result.

    Args:
        test_filename (Optional[str]): The name of the file from which the result was extracted.
        test_date (Optional[datetime]): The date and time the test was conducted.
        test_common_name (Optional[str]): A standardized or simplified name for the test.
        test_name (Optional[str]): The raw or detailed name of the test.
        test_result (Optional[float]): The numerical result of the test.
        test_uom (Optional[str]): The unit of measurement used for the test result.
        classification (Optional[str]): Classification such as 'high', 'low', or 'normal'.
        reason (Optional[str]): Explanation or reasoning behind the classification.
        recommendation (Optional[str]): Medical or practical recommendations based on the result.
    """    
    def __init__(
        self,
        test_filename: Optional[str] = None,
        test_date: Optional[datetime] = None,
        test_common_name: Optional[str] = None,
        test_name: Optional[str] = None,
        test_result: Optional[float] = None,
        test_uom: Optional[str] = None,
        classification: Optional[str] = None,
        reason: Optional[str] = None,
        recommendation: Optional[str] = None,
    ):
        self.test_filename = test_filename
        self.test_date = test_date
        self.test_common_name = test_common_name
        self.test_name = test_name
        self.test_result = test_result
        self.test_uom = test_uom
        self.classification = classification
        self.reason = reason
        self.recommendation = recommendation

    @classmethod
    def from_dict(
        cls,
        data: Dict[str, Optional[str]],
        filename: Optional[str] = None,
        test_date: Optional[str] = None
    ) -> "LabResult":
        """
        Create a LabResult instance from a dictionary of test data.

        Args:
            data (Dict[str, Optional[str]]): A dictionary containing keys corresponding to lab result fields.
            filename (Optional[str]): The name of the file where the lab result was found.
            test_date (Optional[str]): An optional override for the test date. If not provided, data["test_date"] is used.

        Returns:
            LabResult: An instance of the LabResult class populated with the provided data.
        """
        return cls(
            test_filename= filename,
            test_date= test_date or data.get("test_date"),
            test_common_name=data.get("test_comon_name"),
            test_name=data.get("test_name"),
            test_result=data.get("test_result"),
            test_uom=data.get("test_uom"),
            classification=data.get("classification"),
            reason=data.get("reason"),
            recommendation=data.get("recommendation"),
        )

    def get_datetime_object(self) -> Optional[date]:
        """
        Converts the test_date string into a datetime.date object.

        Expected format: "11 Jan 2025, 08:04 AM"

        Returns:
            Optional[date]: A date object if parsing succeeds, otherwise None.
        """
        if self.test_date:
            try:
                return datetime.strptime(self.test_date, "%d %b %Y, %I:%M %p").date()
            except ValueError:
                pass  # You could log the error if needed
        return None
    
class LabResultList:
    """
    A container class for holding and manipulating a list of LabResult objects.
    """

    def __init__(self):
        """
        Initializes an empty list to store LabResult instances.
        """
        self.result: List[LabResult] = []

    def extend(self, lab_result_list: "LabResultList") -> None:
        """
        Extends the current LabResult list with another LabResultList.

        Args:
            lab_result_list (LabResultList): Another instance whose results will be appended.
        """
        self.result.extend(lab_result_list.result)

    def get_unique_test_names_str(self) -> str:
        """
        Returns a newline-separated string of unique (test_common_name, test_name) pairs.
        """
        unique_pairs = {
            (result.test_common_name, result.test_name)
            for result in self.result
        }
        return "\n".join(f"{common or ''} -> {name or ''}" for common, name in unique_pairs)
    
    def get_unmapped_test_names_str(self) -> str:
        """
        Returns a newline-separated string of LabResult.test_name values
        where test_common_name is None.

        Returns:
            str: A string of test_name entries with no common name, separated by newlines.
        """
        return "\n".join(
            result.test_name for result in self.result
            if result.test_common_name is None and result.test_name
        ) 

    def apply_standardization(self, correction_dict: Dict[str, str]) -> None:
        """
        Update test_common_name for each LabResult based on correction_dict,
        using test_name as the key.
        """
        for result in self.result:
            if result.test_name and result.test_name in correction_dict:
                result.test_common_name = correction_dict[result.test_name]

    def describe(self) -> str:
        """
        Returns a formatted string listing each LabResult in the list
        with all its attributes and values, along with the total count.
        """
        descriptions = [f"Total Lab Results: {len(self.result)}"]
        for idx, result in enumerate(self.result, start=1):
            lines = [f"\nLabResult #{idx}"]
            for attr, value in vars(result).items():
                lines.append(f"  {attr}: {value}")
            descriptions.append("\n".join(lines))
        return "\n".join(descriptions)
import json
from datetime import date, datetime
from typing import Optional, List, Dict, Set, Tuple

from src.utils.llm_client import LLMClient

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


class LabResultList:
    """
    A container class for holding and manipulating a list of LabResult objects.
    """

    def __init__(self):
        """
        Initializes an empty list to store LabResult instances.
        """
        self.result: List[LabResult] = []


    def get_unique_test_names_str(self) -> str:
        """
        Constructs a newline-separated string of unique (test_common_name, test_name) pairs
        from the lab results.

        Returns:
            str: A string where each line represents a unique pair in the format 
                'test_common_name -> test_name'. If any field is None, it is replaced with an empty string.
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

        This is typically used to identify test names that need standardization.

        Returns:
            str: A newline-separated string of test_name values lacking a standard name.
        """
        return "\n".join(
            result.test_name for result in self.result
            if result.test_common_name is None and result.test_name
        ) 
    

    def apply_standardization(self, correction_dict: Dict[str, str]) -> None:
        """
        Applies standardization to each LabResult's `test_common_name` based on a correction dictionary.

        Args:
            correction_dict (Dict[str, str]): A mapping from variant `test_name` to standardized `test_common_name`.
        """
        for result in self.result:
            if result.test_name and result.test_name in correction_dict:
                result.test_common_name = correction_dict[result.test_name]


    def describe(self) -> str:
        """
        Returns a formatted string summarizing all LabResult entries in the list.
        Includes the total count and all attributes of each LabResult instance.

        Returns:
            str: A human-readable summary of the lab results.
        """
        descriptions = [f"Total Lab Results: {len(self.result)}"]
        for idx, result in enumerate(self.result, start=1):
            lines = [f"\nLabResult #{idx}"]
            for attr, value in vars(result).items():
                lines.append(f"  {attr}: {value}")
            descriptions.append("\n".join(lines))
        return "\n".join(descriptions)
    
    
    def standardize_test_names(
        self, settings_dict: dict, prompt_template: str, unique_name_pairs: str
    ) -> None:
        """
        Uses an LLM to standardize test names in the LabResultList by updating test_common_name.

        Args:
            settings_dict (dict): Dictionary of LLM settings and API keys.
            prompt_template (str): Template used to generate the LLM prompt.
            unique_name_pairs (str): String of existing (common_name -> test_name) mappings.
        """
        unmapped_names = self.get_unmapped_test_names_str()
        if not unmapped_names.strip():
            return  # Nothing to standardize

        response = LLMClient.run_prompt(
            settings_dict=settings_dict,
            prompt_template=prompt_template,
            prompt_context={
                "standard_mappings": unique_name_pairs,
                "new_variants": unmapped_names
            }
        )

        try:
            classified_data = json.loads(response)
            correction_dict = {
                item["variant_name"]: item["standard_name"]
                for item in classified_data
            }
            self.apply_standardization(correction_dict)
        except (json.JSONDecodeError, KeyError) as e:
            raise ValueError(f"Standardization failed: {e}")
        
from datetime import datetime
from typing import Optional, List

class cls_Lab_Result:
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
    def from_dict(cls, data: dict, filename:Optional[str]=None,
                  test_date: Optional[str] = None) -> "cls_Lab_Result":
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

    def get_datetime_object(self) -> Optional[datetime]:
        if self.test_date:
            try:
                return datetime.strptime(self.test_date, "%d %b %Y, %I:%M %p").date()
            except ValueError:
                pass
        return None
    
class cls_Lab_Result_List():
    def __init__(self):
        self.result: List[cls_Lab_Result] = []


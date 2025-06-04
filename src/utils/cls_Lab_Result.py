from datetime import datetime
from typing import Optional

class cls_Lab_Result:
    def __init__(
        self,
        test_datetime: Optional[str] = None,
        test_name: Optional[str] = None,
        test_result: Optional[float] = None,
        test_uom: Optional[str] = None,
        ref_range: Optional[str] = None,
        diagnostic: Optional[str] = None,
    ):
        self.test_datetime = test_datetime
        self.test_name = test_name
        self.test_result = test_result
        self.test_uom = test_uom
        self.ref_range = ref_range
        self.diagnostic = diagnostic

    @classmethod
    def from_dict(cls, data: dict) -> "cls_Lab_Result":
        return cls(
            test_datetime=data.get("datetime"),
            test_name=data.get("test_name"),
            test_result=data.get("test_result"),
            test_uom=data.get("test_uom"),
            ref_range=data.get("ref_range"),
            diagnostic=data.get("diagnostic"),
        )

    def get_datetime_object(self) -> Optional[datetime]:
        if self.test_datetime:
            try:
                return datetime.strptime(self.test_datetime, "%d %b %Y, %I:%M %p")
            except ValueError:
                pass
        return None
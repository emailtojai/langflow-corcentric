import requests
from datetime import datetime
from langflow.custom import Component
from langflow.io import MessageTextInput, Output
from langflow.schema import Data
from langchain_core.tools import tool

class StockCheckTool(Component):
    display_name = "Inventory Check Tool"
    description = "Checks inventory availability for a part, quantity, and fulfillment date."
    icon = "Package"  # Suitable icon for stock/inventory
    name = "StockCheckTool"

    inputs = [
        MessageTextInput(
            name="input_str",
            display_name="Stock Check Input",
            info="Input in the format: buyer_part_number=\"some part\", order_quantity=\"some qty\", requested_fulfillment_date=\"MM/DD/YY or YYYY\" (e.g., buyer_part_number=\"3010228002\", order_quantity=\"32100.000\", requested_fulfillment_date=\"02/13/2025\").",
            value="",
            tool_mode=True,  # Enable Tool Mode toggle
        ),
    ]

    outputs = [
        Output(
            display_name="Result",
            name="result",
            method="build",
        ),
    ]

    def build(self, **kwargs) -> Data:
        """
        Execute the stock check logic and return the result as a Data object.
        Accepts input_str either directly or from kwargs.
        """
        # Extract input_str from kwargs or use default
        input_str = kwargs.get("input_str", self.input_str).strip()

        # 1. Split the string by commas -> should get exactly 3 segments
        parts = input_str.split(",")
        if len(parts) != 3:
            result = {"error": "Expected exactly 3 comma-separated segments. Format: buyer_part_number=\"...\", order_quantity=\"...\", requested_fulfillment_date=\"...\""}
            self.status = result["error"]
            return Data(data=result)

        # Prepare placeholders
        buyer_part_number = None
        order_quantity = None
        requested_fulfillment_date = None

        # 2. Parse each segment: 'key="value"'
        for part in parts:
            sub_parts = part.strip().split("=", maxsplit=1)
            if len(sub_parts) != 2:
                result = {"error": f"Could not parse segment: {part}"}
                self.status = result["error"]
                return Data(data=result)

            key = sub_parts[0].strip()
            raw_value = sub_parts[1].strip().strip('"')  # Remove surrounding quotes

            if key == "buyer_part_number":
                buyer_part_number = raw_value
            elif key == "order_quantity":
                order_quantity = raw_value
            elif key == "requested_fulfillment_date":
                requested_fulfillment_date = raw_value
            else:
                result = {"error": f"Unrecognized key '{key}' in segment: {part}"}
                self.status = result["error"]
                return Data(data=result)

        # 3. Validate buyer_part_number
        if not buyer_part_number or not buyer_part_number.strip():
            result = {"error": "'buyer_part_number' must be a non-empty string."}
            self.status = result["error"]
            return Data(data=result)

        # 4. Validate order_quantity (float -> int)
        try:
            quantity_float = float(order_quantity)
            order_quantity_int = int(quantity_float)
            if order_quantity_int <= 0:
                result = {"error": "'order_quantity' must be a positive integer."}
                self.status = result["error"]
                return Data(data=result)
        except ValueError:
            result = {"error": "'order_quantity' must be a valid integer (e.g., '32100', '32100.000')."}
            self.status = result["error"]
            return Data(data=result)

        # 5. Validate requested_fulfillment_date
        requested_date_obj = None
        tried_formats = ["%m/%d/%y", "%m/%d/%Y"]
        success = False
        for fmt in tried_formats:
            try:
                requested_date_obj = datetime.strptime(requested_fulfillment_date, fmt).date()
                success = True
                break
            except ValueError:
                pass

        if not success:
            result = {"error": "'requested_fulfillment_date' must be in 'MM/DD/YY' or 'MM/DD/YYYY' format (e.g., '5/1/25' or '5/1/2025')."}
            self.status = result["error"]
            return Data(data=result)

        # 6. Make the API call
        url = "http://192.168.1.248:5001/check_stock"
        params = {
            "buyer_part_number": buyer_part_number.strip(),
            "order_quantity": order_quantity_int,
            "requested_fulfillment_date": requested_date_obj.strftime("%Y-%m-%d"),
        }

        try:
            response = requests.post(url, params=params, timeout=5)
            response.raise_for_status()
            data = response.json()
            restock_info = (
                f" Expected restock date: {data['expected_restock_date']}"
                if data.get("expected_restock_date")
                else ""
            )
            if data.get("ItemsInStock"):
                message = (
                    f"Stock is available for {buyer_part_number} "
                    f"(Quantity: {order_quantity_int}).{restock_info}"
                )
            else:
                message = (
                    f"Stock is NOT available for {buyer_part_number} "
                    f"(Quantity: {order_quantity_int}).{restock_info}, "
                    "please ask_human on how to proceed"
                )
            result = {
                "buyer_part_number": buyer_part_number,
                "order_quantity": order_quantity_int,
                "requested_fulfillment_date": requested_date_obj.strftime("%Y-%m-%d"),
                "message": message
            }
            display_result = message
        except requests.exceptions.RequestException as e:
            result = {"error": f"Stock check failed: {str(e)}"}
            display_result = f"Error: {str(e)}"

        self.status = display_result
        return Data(data=result)

    def process_input(self) -> str:
        """Callable method for tool execution."""
        result = self.build()
        return result.data.get("message", result.data.get("error", "No result"))

@tool
def check_stock_service(input_str: str) -> str:
    """
    LangChain-compatible tool version of the stock check service.
    """
    instance = StockCheckTool(input_str=input_str)
    return instance.process_input()
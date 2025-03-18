import requests
from langflow.custom import Component
from langflow.io import MessageTextInput, Output
from langflow.schema import Data
from langchain_core.tools import tool

class PriceCheckTool(Component):
    display_name = "Price Check Tool"
    description = "Checks price for a given buyer part number against a PO price."
    icon = "DollarSign"  # Choose an appropriate icon
    name = "PriceCheckTool"

    inputs = [
        MessageTextInput(
            name="input_str",
            display_name="Price Check Input",
            info="Input in the format: buyer_part_number=\"some part\", po_price=\"some price\" (e.g., buyer_part_number=\"3010228002\", po_price=\"125.50\").",
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
        Execute the price check logic and return the result as a Data object.
        Accepts input_str either directly or from kwargs.
        """
        # Extract input_str from kwargs or use default
        input_str = kwargs.get("input_str", self.input_str).strip()

        # 1. Split the string by comma -> should get exactly 2 segments
        parts = input_str.split(",")
        if len(parts) != 2:
            result = {"error": "Expected exactly 2 comma-separated segments. Format: buyer_part_number=\"...\", po_price=\"...\""}
            self.status = result["error"]
            return Data(data=result)

        buyer_part_number = None
        po_price_str = None

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
            elif key == "po_price":
                po_price_str = raw_value
            else:
                result = {"error": f"Unrecognized key '{key}' in segment: {part}"}
                self.status = result["error"]
                return Data(data=result)

        # 3. Validate buyer_part_number
        if not buyer_part_number or not buyer_part_number.strip():
            result = {"error": "'buyer_part_number' must be a non-empty string."}
            self.status = result["error"]
            return Data(data=result)

        # 4. Validate po_price (float)
        try:
            po_price_float = float(po_price_str)
        except ValueError:
            result = {"error": "'po_price' must be a valid number (e.g., '125.50')."}
            self.status = result["error"]
            return Data(data=result)

        # 5. Make the API call to /check_price
        url = "http://192.168.1.248:5001/check_price"
        params = {
            "buyer_part_number": buyer_part_number.strip(),
            "po_price": po_price_float,
        }

        try:
            response = requests.post(url, params=params, timeout=5)  # Add timeout
            response.raise_for_status()
            data = response.json()
            message = data.get("message", "No message returned from /check_price endpoint.")
            result = {
                "buyer_part_number": buyer_part_number,
                "po_price": po_price_float,
                "message": message
            }
            display_result = f"Price check for {buyer_part_number}: {message}"
        except requests.exceptions.RequestException as e:
            result = {"error": f"Price check failed: {str(e)}"}
            display_result = f"Error: {str(e)}"

        self.status = display_result
        return Data(data=result)

    def process_input(self) -> str:
        """Callable method for tool execution."""
        result = self.build()
        return result.data.get("message", result.data.get("error", "No result"))

@tool
def price_check_service(input_str: str) -> str:
    """
    LangChain-compatible tool version of the price check service.
    """
    instance = PriceCheckTool(input_str=input_str)
    return instance.process_input()
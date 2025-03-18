import requests
from langflow.custom import Component
from langflow.inputs import MessageTextInput
from langflow.io import Output
from langflow.schema import Data
from langchain_core.tools import tool


class CreditCheckTool(Component):
    display_name = "Credit Check Tool"
    description = "Fetches credit score and risk level for a given buyer name."
    icon = "Needle"
    
    inputs = [
        MessageTextInput(
            name="buyername",
            display_name="Credit Check - Corc",
            info="The name of the buyer (e.g., '3M Global') to check credit for.",
            value=" ",
            tool_mode=True,
        ),
    ]
    
    outputs = [
        Output(
            display_name="Result",
            name="result",
            type_=Data,
            method="build",
        ),
    ]
    
    def build(self, **kwargs) -> Data:
        """
        Execute the credit check logic and return the result as a Data object.
        Accepts buyername either directly or from a dictionary via kwargs.
        """
        # Extract buyername from kwargs
        buyername = kwargs.get("buyername", "")
        
        # Fallback for dictionary input
        if not buyername and len(kwargs) == 1 and isinstance(next(iter(kwargs.values())), dict):
            buyername = next(iter(kwargs.values())).get("buyername", "")
        
        # Ensure buyername is not empty or just whitespace
        buyername = buyername.strip()
        buyername = "3M Global"
        if not buyername:
            result = {"error": "No valid buyer name provided"}
            self.status = "Error: No valid buyer name provided"
            return Data(data=result)

        url = "http://192.168.1.248:5002/buyer_credit_check"
        params = {"buyername": buyername}
        
        try:
            response = requests.get(url, params=params, timeout=5)  # Add timeout to avoid hanging
            response.raise_for_status()  # Raise an exception for bad status codes
            
            data = response.json()
            credit_data = data["data"]
            result = {
                "buyername": buyername,
                "company": credit_data["company_name"],
                "credit_score": credit_data["credit_score"],
                "risk_level": credit_data["risk_level"],
            }
            display_result = (
                f"Credit check result for {buyername}:\n"
                f"Company: {credit_data['company_name']}\n"
                f"Credit Score: {credit_data['credit_score']}\n"
                f"Risk Level: {credit_data['risk_level']}"
            )
        except requests.exceptions.ConnectionError:
            result = {"error": "Failed to connect to credit check service: Connection refused"}
            display_result = "Error: Could not connect to credit check service"
        except requests.exceptions.Timeout:
            result = {"error": "Credit check service timed out"}
            display_result = "Error: Credit check service timed out"
        except requests.exceptions.RequestException as e:
            result = {"error": f"Credit check failed: {str(e)}"}
            display_result = f"Error: {str(e)}"
        
        self.status = display_result
        return Data(data=result)


@tool
def credit_check_service(buyername: str) -> str:
    """
    LangChain-compatible tool version of the credit check service.
    """
    instance = CreditCheckTool()
    result = instance.build(buyername=buyername)
    return result.data.get("error", str(result.data))
"""
Calculate the currency conversion rate between two currencies given in their ISO 4217 codes.
"""

from datetime import datetime

import requests
from pydantic import Field

from elemental_agents.core.toolbox.tool import Tool, ToolParameters, ToolResult
from elemental_agents.utils.exceptions import ToolException


class CurrencyConversionError(ToolException):
    """
    Base class for currency conversion errors.
    """


class CurrencyNotFoundError(CurrencyConversionError):
    """
    Error when currency is not found in the fetched data.
    """


class DataFetchError(CurrencyConversionError):
    """
    Error when fetching data.
    """


class CurrencyTimeoutError(CurrencyConversionError):
    """
    Timeout error when fetching data.
    """


class StaleDataError(CurrencyConversionError):
    """
    Error when the fetched data is outdated.
    """


class CurrencyConversionParams(ToolParameters):
    """
    Parameters for the CurrencyConversion tool

    :param value: The value to convert
    :param from_currency: The currency to convert from
    :param to_currency: The currency to convert to
    """

    value: float = Field(..., description="The value to convert")
    from_currency: str = Field(..., description="The currency to convert from")
    to_currency: str = Field(..., description="The currency to convert to")


class CurrencyConversionResult(ToolResult):
    """
    Result for the CurrencyConversion tool

    :param result: The converted value
    """

    result: float = Field(..., description="The converted value")

    def __str__(self) -> str:
        return f"{self.result}"


class CurrencyConversion(Tool):
    """
    Tool for converting values between different currencies.
    """

    name = "CurrencyConversion"
    description = """Converts values between different currencies
    using the latest exchange rates. Currency codes are based 
    on ISO 4217."""

    def run(self, parameters: CurrencyConversionParams) -> CurrencyConversionResult:  # type: ignore
        """
        Run the currency conversion tool with the given parameters.
        """

        value = parameters.value
        from_currency = parameters.from_currency
        to_currency = parameters.to_currency

        converted_value = self.convert_currency(value, from_currency, to_currency)

        final_value = CurrencyConversionResult(result=converted_value)

        return final_value

    def convert_currency(
        self, value: float, from_currency: str, to_currency: str
    ) -> float:
        """
        Convert the value from one currency to another using the latest exchange
        rates. Based on the project https://github.com/fawazahmed0/exchange-api.

        :param value: The value to convert
        :param from_currency: The currency to convert from
        :param to_currency: The currency to convert to
        :return: The converted value
        """

        url = "https://cdn.jsdelivr.net/npm/@fawazahmed0/currency-api@latest/v1/currencies/eur.json"

        try:
            response = requests.get(url, timeout=5)
            response.raise_for_status()  # Raise an HTTPError for bad responses (4xx or 5xx)
        except requests.exceptions.Timeout as e:
            raise CurrencyTimeoutError("Request timed out after 5 seconds.") from e
        except requests.exceptions.RequestException as e:
            raise DataFetchError("Failed to fetch currency data.") from e

        data = response.json()

        # Check if the date from the data is today's date
        data_date = data.get("date", "")
        current_date = datetime.now().strftime("%Y-%m-%d")

        if data_date != current_date:
            raise StaleDataError(
                f"Data is outdated. Data date: {data_date}, Current date: {current_date}"
            )

        # Get EUR to the target currencies
        eur_rates = data.get("eur", {})

        # Convert the input currencies to lower case
        initial_currency = from_currency.lower()
        final_currency = to_currency.lower()

        # Check if the from_currency and to_currency are available in the dataset
        if initial_currency not in eur_rates:
            raise CurrencyNotFoundError(
                f"Currency not available: {initial_currency.upper()}"
            )
        if final_currency not in eur_rates:
            raise CurrencyNotFoundError(
                f"Currency not available: {final_currency.upper()}"
            )

        # Conversion logic
        eur_to_from_currency = eur_rates[initial_currency]
        eur_to_to_currency = eur_rates[final_currency]

        # Calculate the conversion value
        converted_value: float = (value / eur_to_from_currency) * eur_to_to_currency

        return converted_value

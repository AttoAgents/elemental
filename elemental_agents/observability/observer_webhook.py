"""
Observability functions to send data to a webhook.
"""

import requests
from loguru import logger

from elemental_agents.observability.observer_data_model import (
    ObserverInteraction,
    ObserverMessage,
    ObserverSession,
    ObserverTask,
)


def send_to_webhook(
    webhook_url: str,
    payload_type: str,
    payload: ObserverMessage | ObserverTask | ObserverInteraction | ObserverSession,
) -> None:
    """
    Send the observer data to a webhook.

    :param webhook_url: URL of the webhook to send the data.
    :param payload_type: Type of the payload. Can be "message", "task", "interaction", or "session".
    :param payload: Payload data to send to the webhook.
    """
    logger.info(f"Sending observer data to webhook. Type: {payload_type}.")

    # Send the payload to the webhook
    response = requests.post(
        url=webhook_url,
        json={"payload_type": payload_type, "payload": payload.model_dump_json()},
        timeout=5,
    )

    if response.status_code == 200:
        logger.info("Observer data sent to webhook successfully.")
    else:
        logger.error(
            "Failed to send observer data to webhook."
            f"Status code: {response.status_code}"
        )

#!/usr/bin/env python
# This is an example of sending a slack notification. For more details see
# official docs:
# https://api.slack.com/messaging/webhooks

import requests
import json

# This URL is tied to a single channel. That can be generalized, or you can
# create a new "app" to use another channel.
WEBHOOK = (
    "https://hooks.slack.com/services/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"
)


def notify(message):
    headers = {"Content-Type": "application/json"}
    data = json.dumps({"text": message})
    requests.post(WEBHOOK, data=data, headers=headers)


if __name__ == "__main__":
    print("Please type your message.")
    message = input("message> ")
    notify(message)

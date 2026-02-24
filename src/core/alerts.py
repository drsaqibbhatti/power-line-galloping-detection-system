from __future__ import annotations
import requests

def send_webhook(url: str, payload: dict) -> tuple[bool, str]:
    try:
        r = requests.post(url, json=payload, timeout=5)
        return (200 <= r.status_code < 300), f"{r.status_code}"
    except Exception as e:
        return False, str(e)

def send_telegram(bot_token: str, chat_id: str, text: str) -> tuple[bool, str]:
    try:
        url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
        r = requests.post(url, json={"chat_id": chat_id, "text": text}, timeout=5)
        return (200 <= r.status_code < 300), f"{r.status_code}"
    except Exception as e:
        return False, str(e)

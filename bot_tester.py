#!/usr/bin/env python3
"""
Automated Bot Tester for poemai Chatbots
Extracts the testing functionality from deploy_case_manager_and_auto_test.py
"""

import argparse
import json
import logging
import os
import time
from unittest import result
import uuid
import zoneinfo
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from decimal import Decimal
from enum import Enum
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

import boto3
import requests
import yaml
from jinja2 import Template
from poemai_utils.basic_types_utils import (any_to_bool, linebreak,
                                            replace_decimal_with_string,
                                            replace_floats_with_decimal)
from poemai_utils.enum_utils import add_enum_attrs, add_enum_repr
from poemai_utils.openai.ask import Ask
from poemai_utils.time_utils import current_time_iso
from pydantic import BaseModel

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
_logger = logging.getLogger(__name__)


class LanaguageCode(str, Enum):
    de = "de"  # German
    tr = "tr"  # Turkish
    uk = "uk"  # Ukrainian
    ar = "ar"  # Arabic
    prs = "prs"  # Dari
    ti = "ti"  # Tigrinya
    fa = "fa"  # Persian (Farsi)
    ru = "ru"  # Russian
    ta = "ta"  # Tamil
    pt = "pt"  # Portuguese
    sq = "sq"  # Albanian
    ps = "ps"  # Pashto
    fr = "fr"  # French
    it = "it"  # Italian
    ku = "ku"  # Kurdish Kurmanji
    es = "es"  # Spanish
    so = "so"  # Somali
    en = "en"  # English
    sr = "sr"  # Serbian
    hu = "hu"  # Hungarian
    ckb = "ckb"  # Kurdish Sorani
    bs = "bs"  # Bosnian
    kmr = "kmr"  # Kurdish Badini
    am = "am"  # Amharic
    hr = "hr"  # Croatian
    th = "th"  # Thai
    bo = "bo"  # Tibetan
    pl = "pl"  # Polish
    zh = "zh"  # Chinese Mandarin
    ln = "ln"  # Lingala
    mk = "mk"  # Macedonian
    arq = "arq"  # Maghrebi Arabic
    uz = "uz"  # Uzbek
    ro = "ro"  # Romanian
    sl = "sl"  # Slovenian
    sw = "sw"  # Swahili
    ur = "ur"  # Urdu
    si = "si"  # Sinhala
    byn = "byn"  # Bilen
    vi = "vi"  # Vietnamese
    tl = "tl"  # Tagalog (Filipino)
    tig = "tig"  # Tigre
    zza = "zza"  # Kurdish Zazaki


add_enum_attrs(
    {
        LanaguageCode.de: {
            "language_name": "German",
            "iso639_1": "de",
            "language_name_de": "Deutsch",
        },
        LanaguageCode.tr: {
            "language_name": "Turkish",
            "iso639_1": "tr",
            "language_name_de": "Türkisch",
        },
        LanaguageCode.uk: {
            "language_name": "Ukrainian",
            "iso639_1": "uk",
            "language_name_de": "Ukrainisch",
        },
        LanaguageCode.ar: {
            "language_name": "Arabic",
            "iso639_1": "ar",
            "language_name_de": "Arabisch",
        },
        LanaguageCode.prs: {
            "language_name": "Dari",
            "iso639_1": "prs",
            "language_name_de": "Dari",
        },
        LanaguageCode.ti: {
            "language_name": "Tigrinya",
            "iso639_1": "ti",
            "language_name_de": "Tigrinya",
        },
        LanaguageCode.fa: {
            "language_name": "Persian (Farsi)",
            "iso639_1": "fa",
            "language_name_de": "Persisch (Farsi)",
        },
        LanaguageCode.ru: {
            "language_name": "Russian",
            "iso639_1": "ru",
            "language_name_de": "Russisch",
        },
        LanaguageCode.ta: {
            "language_name": "Tamil",
            "iso639_1": "ta",
            "language_name_de": "Tamil",
        },
        LanaguageCode.pt: {
            "language_name": "Portuguese",
            "iso639_1": "pt",
            "language_name_de": "Portugiesisch",
        },
        LanaguageCode.sq: {
            "language_name": "Albanian",
            "iso639_1": "sq",
            "language_name_de": "Albanisch",
        },
        LanaguageCode.ps: {
            "language_name": "Pashto",
            "iso639_1": "ps",
            "language_name_de": "Paschtu",
        },
        LanaguageCode.fr: {
            "language_name": "French",
            "iso639_1": "fr",
            "language_name_de": "Französisch",
        },
        LanaguageCode.it: {
            "language_name": "Italian",
            "iso639_1": "it",
            "language_name_de": "Italienisch",
        },
        LanaguageCode.ku: {
            "language_name": "Kurdish Kurmanji",
            "iso639_1": "ku",
            "language_name_de": "Kurdisch (Kurmandschi)",
        },
        LanaguageCode.es: {
            "language_name": "Spanish",
            "iso639_1": "es",
            "language_name_de": "Spanisch",
        },
        LanaguageCode.so: {
            "language_name": "Somali",
            "iso639_1": "so",
            "language_name_de": "Somali",
        },
        LanaguageCode.en: {
            "language_name": "English",
            "iso639_1": "en",
            "language_name_de": "Englisch",
        },
        LanaguageCode.sr: {
            "language_name": "Serbian",
            "iso639_1": "sr",
            "language_name_de": "Serbisch",
        },
        LanaguageCode.hu: {
            "language_name": "Hungarian",
            "iso639_1": "hu",
            "language_name_de": "Ungarisch",
        },
        LanaguageCode.ckb: {
            "language_name": "Kurdish Sorani",
            "iso639_1": "ckb",
            "language_name_de": "Kurdisch (Sorani)",
        },
        LanaguageCode.bs: {
            "language_name": "Bosnian",
            "iso639_1": "bs",
            "language_name_de": "Bosnisch",
        },
        LanaguageCode.kmr: {
            "language_name": "Kurdish Badini",
            "iso639_1": "kmr",
            "language_name_de": "Kurdisch (Badini)",
        },
        LanaguageCode.am: {
            "language_name": "Amharic",
            "iso639_1": "am",
            "language_name_de": "Amharisch",
        },
        LanaguageCode.hr: {
            "language_name": "Croatian",
            "iso639_1": "hr",
            "language_name_de": "Kroatisch",
        },
        LanaguageCode.th: {
            "language_name": "Thai",
            "iso639_1": "th",
            "language_name_de": "Thailändisch",
        },
        LanaguageCode.bo: {
            "language_name": "Tibetan",
            "iso639_1": "bo",
            "language_name_de": "Tibetisch",
        },
        LanaguageCode.pl: {
            "language_name": "Polish",
            "iso639_1": "pl",
            "language_name_de": "Polnisch",
        },
        LanaguageCode.zh: {
            "language_name": "Chinese Mandarin",
            "iso639_1": "zh",
            "language_name_de": "Chinesisch (Mandarin)",
        },
        LanaguageCode.ln: {
            "language_name": "Lingala",
            "iso639_1": "ln",
            "language_name_de": "Lingala",
        },
        LanaguageCode.mk: {
            "language_name": "Macedonian",
            "iso639_1": "mk",
            "language_name_de": "Mazedonisch",
        },
        LanaguageCode.arq: {
            "language_name": "Maghrebi Arabic",
            "iso639_1": "arq",
            "language_name_de": "Arabisch (Maghrebi)",
        },
        LanaguageCode.uz: {
            "language_name": "Uzbek",
            "iso639_1": "uz",
            "language_name_de": "Usbekisch",
        },
        LanaguageCode.ro: {
            "language_name": "Romanian",
            "iso639_1": "ro",
            "language_name_de": "Rumänisch",
        },
        LanaguageCode.sl: {
            "language_name": "Slovenian",
            "iso639_1": "sl",
            "language_name_de": "Slowenisch",
        },
        LanaguageCode.sw: {
            "language_name": "Swahili",
            "iso639_1": "sw",
            "language_name_de": "Suaheli",
        },
        LanaguageCode.ur: {
            "language_name": "Urdu",
            "iso639_1": "ur",
            "language_name_de": "Urdu",
        },
        LanaguageCode.si: {
            "language_name": "Sinhala",
            "iso639_1": "si",
            "language_name_de": "Singhalesisch",
        },
        LanaguageCode.byn: {
            "language_name": "Bilen",
            "iso639_1": "byn",
            "language_name_de": "Bilen",
        },
        LanaguageCode.vi: {
            "language_name": "Vietnamese",
            "iso639_1": "vi",
            "language_name_de": "Vietnamesisch",
        },
        LanaguageCode.tl: {
            "language_name": "Tagalog (Filipino)",
            "iso639_1": "tl",
            "language_name_de": "Tagalog (Filipino)",
        },
        LanaguageCode.tig: {
            "language_name": "Tigre",
            "iso639_1": "tig",
            "language_name_de": "Tigre",
        },
        LanaguageCode.zza: {
            "language_name": "Kurdish Zazaki",
            "iso639_1": "zza",
            "language_name_de": "Kurdisch (Zazaki)",
        },
    }
)


class ApiConfig(BaseModel):
    base_url: str


class Scenario(BaseModel):
    name: str
    situation: str
    check_template: str = None
    max_turns: int = 20
    check_instructions: str = None
    scenario_languages: list[LanaguageCode] = [LanaguageCode.de]
    skip: Optional[bool] = False


class Config(BaseModel):
    api: ApiConfig
    corpus_key: str
    model: str = "GPT_4_o_CHATGPT_LATEST"
    prompt_template: str
    check_template: str = None
    max_turns: int = 20
    scenarios: list[Scenario] = []
    test_set_title: Optional[str] = None


class BotTestResultStatus(str, Enum):
    OK = "OK"
    NOK = "NOK"
    SKIPPED = "SKIPPED"


class BotTestResult(BaseModel):
    test_passed: BotTestResultStatus
    description: str


class ActionDurationMeasurement(BaseModel):
    action_name: str
    action_duration: float


class BotTestResultRecord(BaseModel):
    test_name: str
    test_result: BotTestResult
    test_time: str
    test_case_id: str
    test_case_description: str
    test_check_instructions: Optional[str] = None
    case_manager_id: str
    managed_case_id: str
    corpus_key: str
    test_case_language: LanaguageCode
    action_durations: list[ActionDurationMeasurement] = []
    debug_view_url: Optional[str] = None
    conversation_view_url: Optional[str] = None


class BotTestResultsReport(BaseModel):
    test_results: list[BotTestResultRecord] = []
    test_set_title: str = ""
    test_run_start_time: str
    report_url: Optional[str] = None


def load_config(path):
    """Load test configuration from YAML file"""
    with open(path, "r", encoding="utf-8") as f:
        config_data = yaml.safe_load(f)
        try:
            return Config(**config_data)
        except Exception as e:
            _logger.error(f"Error validating config: {e}")
            raise


def make_url_base(api_cfg, corpus_key):
    """Create base URL for API calls"""
    return f"{api_cfg.base_url.rstrip('/')}/{corpus_key}"


def create_new_case(url_base, headers, test_name):
    """Create a new case for testing"""
    _logger.info(
        f"{test_name:<30}: Creating new case; Getting corpus metadata from {url_base}"
    )
    r = requests.get(url_base)
    r.raise_for_status()
    data = r.json()
    cm_id = data["ui_settings"]["case_manager"]["case_manager_default_case_manager_id"]
    url = f"{url_base}/case_managers/{cm_id}/managed_cases"
    r2 = requests.post(url, headers=headers, json={})
    r2.raise_for_status()
    case = r2.json()
    _logger.info(f"New case: {case['managed_case_id']}")
    return case


def get_case_conversation(url_base, case):
    """Get conversation for a case"""
    cm = case["case_manager_id"]
    mc = case["managed_case_id"]
    url = f"{url_base}/case_managers/{cm}/managed_cases/{mc}/conversation"
    r = requests.get(url)
    r.raise_for_status()
    return r.json()


def is_completed(conv):
    """Check if conversation is completed"""
    return conv.get("case_state") == "CASE_COMPLETED"


def poll_until_user_turn(
    url_base, case, headers, interval=0.5, max_retries=60, test_name=""
):
    """Poll until it's the user's turn or case is completed"""
    for _ in range(max_retries):
        conv = get_case_conversation(url_base, case)
        if conv["case_state"] == "WAITING_FOR_USER_INPUT" or is_completed(conv):
            return conv
        _logger.info(f"{test_name:<30}: Waiting for bot response...")
        time.sleep(interval)
    raise RuntimeError("Timeout waiting for bot response")


def format_conversation(conv):
    """Format conversation for display"""
    out = []
    for disp in conv["conversations_list"]:
        for item in disp["conversation_items"]:
            # Handle missing fields gracefully
            role = item.get("display_role", "UNKNOWN")
            content = item.get("content", "")
            # Only add items that have both role and content
            if role != "UNKNOWN" and content:
                out.append(f"{role}: {content}")
    return "\n".join(out)


def last_assistant_message(conv):
    """Get the last assistant message from conversation"""
    assistant_messages = []
    for disp in conv["conversations_list"]:
        for item in disp["conversation_items"]:
            if item["display_role"] == "ASSISTANT":
                assistant_messages.append(item["content"])
    if assistant_messages:
        return assistant_messages[-1]
    return None


def send_user_message(url_base, case, headers, corpus_key, message):
    """Send a user message to the bot"""
    conv = get_case_conversation(url_base, case)
    if conv["case_state"] != "WAITING_FOR_USER_INPUT":
        raise RuntimeError("Not user's turn")
    cid = conv["current_conversation_id"]
    cm = case["case_manager_id"]
    mc = case["managed_case_id"]
    url = f"{url_base}/case_managers/{cm}/managed_cases/{mc}/user_messages"
    payload = {
        "corpus_key": corpus_key,
        "conversation_id": cid,
        "content": message,
        "display_role": "USER",
        "transaction_id": uuid.uuid4().hex[:8],
        "case_manager_id": cm,
        "managed_case_id": mc,
    }

    num_retries_left = 5
    grace_status_codes = (429, 504)  # Too Many Requests, Gateway Timeout
    while True:
        r = requests.post(url, headers=headers, json=payload)
        if r.status_code in grace_status_codes:
            if num_retries_left > 0:
                _logger.warning(
                    f"Rate limited, retrying in 2 seconds... ({num_retries_left} retries left)"
                )
                time.sleep(2)
            else:
                break
        else:
            break
        num_retries_left -= 1
        if num_retries_left <= 0:
            _logger.error("Max retries exceeded for rate limiting")
    r.raise_for_status()
    return r.json()


def run_scenario(scn, cfg, ask, corpus_key):
    """Run a test scenario"""
    url_base = make_url_base(cfg.api, corpus_key)
    test_results = []
    for language in [LanaguageCode(l) for l in scn.scenario_languages]:
        action_durations = []
        headers = {"Accept-Language": f"{language.value};q=0.9"}
        case = None
        turn = 0
        test_result = None

        # Create preliminary BotTestResultRecord with minimal info
        test_name = scn.name
        preliminary_record = BotTestResultRecord(
            test_name=test_name,
            test_result=BotTestResult(
                test_passed=BotTestResultStatus.NOK,
                description="Test not completed.",
            ),
            test_time=current_time_iso(),
            test_case_id="",
            test_case_description=scn.situation,
            test_check_instructions=scn.check_instructions,
            test_case_language=language,
            case_manager_id="",
            managed_case_id="",
            action_durations=[],
            corpus_key=corpus_key,
        )
        _logger.info(
            f"{test_name:<30}: test_case_description: {preliminary_record.test_case_description}"
        )
        if scn.skip:
            preliminary_record.test_result = BotTestResult(
                test_passed=BotTestResultStatus.SKIPPED,
                description="Test skipped as configured.",
            )
            test_results.append(preliminary_record)
            continue

        try:
            # Create case
            start_time = time.time()
            case = create_new_case(url_base, headers, test_name)
            action_durations.append(
                ActionDurationMeasurement(
                    action_name="create_case", action_duration=time.time() - start_time
                )
            )

            preliminary_record.case_manager_id = case["case_manager_id"]
            preliminary_record.managed_case_id = case["managed_case_id"]

            # Initial conversation state
            start_time = time.time()
            conv = poll_until_user_turn(url_base, case, headers, test_name=test_name)
            action_durations.append(
                ActionDurationMeasurement(
                    action_name="initial_poll", action_duration=time.time() - start_time
                )
            )

            # Send initial user message
            start_time = time.time()
            send_user_message(url_base, case, headers, corpus_key, scn.situation)
            action_durations.append(
                ActionDurationMeasurement(
                    action_name="send_initial_message",
                    action_duration=time.time() - start_time,
                )
            )

            turn += 1

            # Conversation loop
            while turn < scn.max_turns:
                start_time = time.time()
                conv = poll_until_user_turn(
                    url_base, case, headers, test_name=test_name
                )
                action_durations.append(
                    ActionDurationMeasurement(
                        action_name=f"poll_turn_{turn}",
                        action_duration=time.time() - start_time,
                    )
                )

                if is_completed(conv):
                    _logger.info(f"{test_name:<30}: Case completed after {turn} turns")
                    break

                # Get last assistant message for evaluation
                last_msg = last_assistant_message(conv)
                if not last_msg:
                    _logger.warning(f"{test_name:<30}: No assistant message found")
                    break

                # If we have a check template, use AI to evaluate
                if scn.check_template or cfg.check_template:
                    check_template = scn.check_template or cfg.check_template
                    formatted_conversation = format_conversation(conv)

                    prompt = check_template.format(
                        situation=scn.situation,
                        conversation=formatted_conversation,
                        check_instructions=scn.check_instructions or "",
                    )

                    start_time = time.time()
                    response = ask.ask(prompt)
                    action_durations.append(
                        ActionDurationMeasurement(
                            action_name=f"ai_check_turn_{turn}",
                            action_duration=time.time() - start_time,
                        )
                    )

                    if "SUCCESS" in response.upper() or "OK" in response.upper():
                        test_result = BotTestResult(
                            test_passed=BotTestResultStatus.OK,
                            description=f"Test passed after {turn} turns. AI evaluation: {response}",
                        )
                        break
                    elif turn >= scn.max_turns - 1:
                        test_result = BotTestResult(
                            test_passed=BotTestResultStatus.NOK,
                            description=f"Test failed - max turns reached. AI evaluation: {response}",
                        )
                        break
                    else:
                        # Continue conversation based on AI feedback
                        continue_prompt = f"Based on the bot's response, continue the conversation to test the scenario. Previous response: {last_msg}"
                        start_time = time.time()
                        next_message = ask.ask(continue_prompt)
                        action_durations.append(
                            ActionDurationMeasurement(
                                action_name=f"ai_continue_turn_{turn}",
                                action_duration=time.time() - start_time,
                            )
                        )

                        start_time = time.time()
                        send_user_message(
                            url_base, case, headers, corpus_key, next_message
                        )
                        action_durations.append(
                            ActionDurationMeasurement(
                                action_name=f"send_message_turn_{turn}",
                                action_duration=time.time() - start_time,
                            )
                        )
                else:
                    # Simple completion check without AI
                    if is_completed(conv):
                        test_result = BotTestResult(
                            test_passed=BotTestResultStatus.OK,
                            description=f"Case completed successfully after {turn} turns",
                        )
                        break

                turn += 1

            if not test_result:
                test_result = BotTestResult(
                    test_passed=BotTestResultStatus.NOK,
                    description=f"Test incomplete - reached max turns ({scn.max_turns})",
                )

        except Exception as e:
            _logger.exception(f"{test_name:<30}: Test failed with error: {e}")
            test_result = BotTestResult(
                test_passed=BotTestResultStatus.NOK,
                description=f"Test failed with error: {str(e)}",
            )

        # Update the preliminary record with final results
        preliminary_record.test_result = test_result
        preliminary_record.action_durations = action_durations
        test_results.append(preliminary_record)

    return test_results


def calc_report_html(test_results_report):
    """Generate HTML report from test results"""
    test_results = test_results_report.test_results
    test_set_title = test_results_report.test_set_title
    test_run_start_time_utc_iso = test_results_report.test_run_start_time
    swiss_tz = zoneinfo.ZoneInfo("Europe/Zurich")
    dt_utc = datetime.fromisoformat(test_run_start_time_utc_iso)
    dt_swiss = dt_utc.astimezone(swiss_tz)
    test_run_start_time_local_time_swiss_format = dt_swiss.strftime("%d.%m.%Y %H:%M:%S")

    HTML_TEMPLATE = """<!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Chatbot Test Report</title>
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/css/bootstrap.min.css" rel="stylesheet">
        <style>
            .table td, .table th { vertical-align: middle; }
            pre { white-space: pre-wrap; word-break: break-word; }
        </style>
    </head>
    <body>
        <div class="container mt-5">
            <h1 class="mb-4">Chatbot Test Report ({{ test_set_title }})</h1>
            <p><strong>Test Run Start Time:</strong> {{ test_run_start_time_local_time_swiss_format }}</p>
            <table class="table table-bordered table-striped">
                <thead class="table-light">
                    <tr>
                        <th>Name</th>
                        <th>Result</th>
                        <th>Language</th>
                        <th>Description</th>
                        <th>Debug Link</th>
                        <th>Conversation Link</th>
                    </tr>
                </thead>
                <tbody>
                    {% for row in test_results %}
                    <tr>
                        <td>{{ row.test_name }}</td>
                        <td>
                            {% if row.test_result.test_passed == "OK" %}
                                <span class="text-success fw-bold">OK</span>
                            {% elif row.test_result.test_passed == "SKIPPED" %}
                                <span class="text-secondary fw-bold">SKIPPED</span>
                            {% else %}
                                <span class="text-danger fw-bold">NOK</span>
                            {% endif %}
                        </td>
                        <td>{{ row.test_case_language }}</td>
                        <td><pre>{{ row.test_result.description }}</pre></td>
                        <td>
                            {% if row.debug_view_url %}
                                <a href="{{ row.debug_view_url }}" target="_blank">Debug</a>
                            {% endif %}
                        </td>
                         <td>
                            {% if row.conversation_view_url %}
                                <a href="{{ row.conversation_view_url }}" target="_blank">Conversation</a>
                            {% endif %}                                
                        </td>
                    </tr>
                    {% endfor %}
                </tbody>
            </table>
        </div>

        <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/js/bootstrap.bundle.min.js"></script>
    </body>
    </html>"""
    tmpl = Template(HTML_TEMPLATE)
    return tmpl.render(
        test_results=test_results,
        test_set_title=test_set_title,
        test_run_start_time_local_time_swiss_format=test_run_start_time_local_time_swiss_format,
    )


def publish_report_to_s3(s3_url, test_results_report):
    """Publish HTML report to S3"""
    if not s3_url.startswith("s3://"):
        raise ValueError(f"Invalid S3 URL: {s3_url}")

    s3_url = s3_url[5:]  # Remove "s3://" prefix
    bucket_name = s3_url.split("/")[0]
    object_key = "/".join(s3_url.split("/")[1:])

    _logger.info(
        f"Publishing test results to S3 bucket {bucket_name} with key {object_key}"
    )

    report_html = calc_report_html(test_results_report)

    # Upload to S3
    s3_client = boto3.client("s3")
    s3_client.put_object(
        Bucket=bucket_name,
        Key=object_key,
        Body=report_html,
        ContentType="text/html",
    )

    _logger.info(f"Test results published to {s3_url}")

    region = os.environ.get("AWS_REGION", "eu-central-2")
    public_url = f"https://{bucket_name}.s3.{region}.amazonaws.com/{object_key}"
    _logger.info(f"\n{'#'*80}\nView the report at: \n{public_url}\n{'#'*80}\n")
    return public_url


def calc_view_urls(
    test_result, debug_url_template=None, conversation_url_template=None
):
    """Calculate view URLs for test results"""
    if not test_result.case_manager_id or not test_result.managed_case_id:
        return None, None

    debug_url = None
    conversation_url = None

    if debug_url_template:
        debug_url = debug_url_template.format(
            corpus_key=test_result.corpus_key,
            case_manager_id=test_result.case_manager_id,
            managed_case_id=test_result.managed_case_id,
        )

    if conversation_url_template:
        conversation_url = conversation_url_template.format(
            corpus_key=test_result.corpus_key,
            case_manager_id=test_result.case_manager_id,
            managed_case_id=test_result.managed_case_id,
        )

    return debug_url, conversation_url


def safe_run_scenario(scn, cfg, ask, corpus_key):
    """Run scenario with error handling"""
    try:
        return run_scenario(scn, cfg, ask, corpus_key)
    except Exception as e:
        _logger.exception(f"Error running scenario {scn.name}: {e}")
        return [
            BotTestResultRecord(
                test_name=scn.name,
                test_result=BotTestResult(
                    test_passed=BotTestResultStatus.NOK,
                    description=f"Scenario failed with error: {str(e)}",
                ),
                test_time=current_time_iso(),
                test_case_id="",
                test_case_description=scn.situation,
                test_check_instructions=scn.check_instructions,
                case_manager_id="",
                managed_case_id="",
                corpus_key=corpus_key,
                test_case_language=LanaguageCode.de,
                action_durations=[],
            )
        ]


def main():
    """Main function"""
    p = argparse.ArgumentParser(description="Automated Chatbot Tester")
    p.add_argument(
        "--config",
        "-c",
        required=True,
        help="Path to the test scenarios / configuration file",
    )
    p.add_argument(
        "--corpus-key",
        "-k",
        required=True,
        help="Corpus key to test against",
    )
    p.add_argument(
        "--results-dir",
        "-r",
        required=False,
        help="Directory to save test results",
    )
    p.add_argument(
        "--publish-s3-url",
        "-p",
        required=False,
        help="S3 URL to publish the test results",
    )
    p.add_argument(
        "--debug-url-template",
        required=False,
        help="URL template for debug links (e.g., 'https://app.staging.poemai.ch/debug/{corpus_key}/{case_manager_id}/{managed_case_id}')",
    )
    p.add_argument(
        "--conversation-url-template",
        required=False,
        help="URL template for conversation links (e.g., 'https://app.staging.poemai.ch/conversation/{corpus_key}/{case_manager_id}/{managed_case_id}')",
    )

    args = p.parse_args()

    results_dir = Path(args.results_dir) if args.results_dir else Path(".")

    cfg = load_config(args.config)
    ask = Ask(
        model=getattr(Ask.OPENAI_MODEL, cfg.model),
        openai_api_key=os.environ.get("OPENAI_API_KEY"),
    )

    test_results_report = BotTestResultsReport(
        test_set_title=cfg.test_set_title or "",
        test_run_start_time=current_time_iso(),
    )

    test_results = []
    with ThreadPoolExecutor(max_workers=4) as executor:
        future_to_scenario = {
            executor.submit(safe_run_scenario, scn, cfg, ask, args.corpus_key): scn
            for scn in cfg.scenarios
        }
        for future in as_completed(future_to_scenario):
            scenario_results = future.result()
            test_results.extend(scenario_results)

    test_results_report.test_results = test_results

    # Add view URLs if templates provided
    for tr in test_results:
        debug_url, conversation_url = calc_view_urls(
            tr, args.debug_url_template, args.conversation_url_template
        )
        tr.debug_view_url = debug_url
        tr.conversation_view_url = conversation_url

    # Publish to S3 if requested
    if args.publish_s3_url:
        public_url = publish_report_to_s3(args.publish_s3_url, test_results_report)
        test_results_report.report_url = public_url

    # Save results to JSON file
    test_results_filename = f"test_results.json"
    with open(results_dir / test_results_filename, "w") as f:
        f.write(
            json.dumps(
                test_results_report.model_dump(),
                default=str,
                ensure_ascii=False,
                indent=4,
            )
        )

    # Print summary
    for tr in test_results:
        status = (
            "✅"
            if tr.test_result.test_passed == BotTestResultStatus.OK
            else "❌" if tr.test_result.test_passed == BotTestResultStatus.NOK else "⏭️"
        )
        _logger.info(
            f"{status} {tr.test_name} ({tr.test_case_language}): {tr.test_result.description}"
        )

    _logger.info(
        "All scenarios completed. Results saved to %s",
        test_results_filename,
    )


if __name__ == "__main__":
    main()

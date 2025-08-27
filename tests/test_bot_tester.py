import json
import logging
import time
from pathlib import Path
from unittest.mock import MagicMock, Mock, mock_open, patch

import pytest
import requests
from pydantic import ValidationError

from bot_tester import (
    ApiConfig,
    BotTestResult,
    BotTestResultRecord,
    BotTestResultsReport,
    BotTestResultStatus,
    Config,
    LanaguageCode,
    Scenario,
    calc_report_html,
    calc_view_urls,
    create_new_case,
    format_conversation,
    get_case_conversation,
    is_completed,
    last_assistant_message,
    load_config,
    make_url_base,
    poll_until_user_turn,
    publish_report_to_s3,
    run_scenario,
    safe_run_scenario,
    send_user_message,
)

_logger = logging.getLogger(__name__)


class TestLanguageCode:
    """Test language code enum functionality"""

    def test_language_codes_exist(self):
        """Test that basic language codes exist"""
        assert LanaguageCode.de == "de"
        assert LanaguageCode.en == "en"
        assert LanaguageCode.fr == "fr"

    def test_language_attributes(self):
        """Test that language attributes are properly set"""
        assert hasattr(LanaguageCode.de, "language_name")
        assert LanaguageCode.de.language_name == "German"
        assert LanaguageCode.de.language_name_de == "Deutsch"


class TestConfigModels:
    """Test Pydantic model validation"""

    def test_api_config_valid(self):
        """Test valid API config"""
        config = ApiConfig(base_url="https://api.example.com")
        assert config.base_url == "https://api.example.com"

    def test_scenario_model_valid(self):
        """Test valid scenario model"""
        scenario = Scenario(
            name="Test Scenario",
            situation="Test situation",
            max_turns=10,
            scenario_languages=[LanaguageCode.de, LanaguageCode.en],
        )
        assert scenario.name == "Test Scenario"
        assert scenario.max_turns == 10
        assert len(scenario.scenario_languages) == 2

    def test_scenario_model_defaults(self):
        """Test scenario model with defaults"""
        scenario = Scenario(name="Test", situation="Test situation")
        assert scenario.max_turns == 20
        assert scenario.scenario_languages == [LanaguageCode.de]
        assert scenario.skip is False

    def test_config_model_valid(self):
        """Test valid config model"""
        config_data = {
            "api": {"base_url": "https://api.example.com"},
            "corpus_key": "test_corpus",
            "model": "GPT_4_o_CHATGPT_LATEST",
            "prompt_template": "Test template",
            "scenarios": [
                {
                    "name": "Test Scenario",
                    "situation": "Test situation",
                }
            ],
        }
        config = Config(**config_data)
        assert config.corpus_key == "test_corpus"
        assert config.model == "GPT_4_o_CHATGPT_LATEST"
        assert len(config.scenarios) == 1

    def test_config_model_defaults(self):
        """Test config model with default values"""
        config_data = {
            "api": {"base_url": "https://api.example.com"},
            "corpus_key": "test_corpus",
            "prompt_template": "Test template",
            "scenarios": [],
        }
        config = Config(**config_data)
        assert config.model == "GPT_4_o_CHATGPT_LATEST"  # Default value
        assert config.max_turns == 20  # Default value
        assert config.test_set_title is None  # Default value

    def test_config_model_invalid(self):
        """Test invalid config model"""
        with pytest.raises(ValidationError):
            Config(api={"base_url": "invalid"})  # Missing required fields

    def test_test_result_record_model(self):
        """Test test result record model"""
        record = BotTestResultRecord(
            test_name="Test",
            test_result=BotTestResult(
                test_passed=BotTestResultStatus.OK, description="Test passed"
            ),
            test_time="2023-01-01T00:00:00Z",
            test_case_id="test_id",
            test_case_description="Test description",
            case_manager_id="cm_id",
            managed_case_id="mc_id",
            corpus_key="test_corpus",
            test_case_language=LanaguageCode.de,
        )
        assert record.test_name == "Test"
        assert record.test_result.test_passed == BotTestResultStatus.OK


class TestConfigLoading:
    """Test configuration loading functionality"""

    def test_load_config_valid_yaml(self):
        """Test loading valid YAML config"""
        yaml_content = """
api:
  base_url: "https://api.example.com"
corpus_key: "test_corpus"
model: "GPT_4_o_CHATGPT_LATEST"
prompt_template: "Test template"
test_set_title: "Test Suite"
scenarios:
  - name: "Scenario 1"
    situation: "Test situation 1"
    max_turns: 15
  - name: "Scenario 2"
    situation: "Test situation 2"
    skip: true
"""
        with patch("builtins.open", mock_open(read_data=yaml_content)):
            config = load_config("test.yaml")
            assert config.corpus_key == "test_corpus"
            assert config.model == "GPT_4_o_CHATGPT_LATEST"
            assert config.test_set_title == "Test Suite"
            assert len(config.scenarios) == 2
            assert config.scenarios[0].max_turns == 15
            assert config.scenarios[1].skip is True

    def test_load_config_invalid_yaml(self):
        """Test loading invalid YAML"""
        invalid_yaml = "invalid: yaml: content: ["
        with patch("builtins.open", mock_open(read_data=invalid_yaml)):
            with pytest.raises(Exception):  # Should raise YAML parsing error
                load_config("test.yaml")

    def test_load_config_missing_required_fields(self):
        """Test loading config with missing required fields"""
        incomplete_yaml = """
api:
  base_url: "https://api.example.com"
# Missing corpus_key and prompt_template but model is included
model: "GPT_4_o_CHATGPT_LATEST"
scenarios: []
"""
        with patch("builtins.open", mock_open(read_data=incomplete_yaml)):
            with pytest.raises(ValidationError):
                load_config("test.yaml")


class TestUrlUtilities:
    """Test URL utility functions"""

    def test_make_url_base(self):
        """Test URL base creation"""
        api_cfg = ApiConfig(base_url="https://api.example.com/")
        corpus_key = "test_corpus"
        result = make_url_base(api_cfg, corpus_key)
        assert result == "https://api.example.com/test_corpus"

    def test_make_url_base_no_trailing_slash(self):
        """Test URL base creation without trailing slash"""
        api_cfg = ApiConfig(base_url="https://api.example.com")
        corpus_key = "test_corpus"
        result = make_url_base(api_cfg, corpus_key)
        assert result == "https://api.example.com/test_corpus"

    def test_calc_view_urls_with_templates(self):
        """Test view URL calculation with templates"""
        test_result = BotTestResultRecord(
            test_name="Test",
            test_result=BotTestResult(
                test_passed=BotTestResultStatus.OK, description="Test"
            ),
            test_time="2023-01-01T00:00:00Z",
            test_case_id="test_id",
            test_case_description="Test",
            case_manager_id="cm_123",
            managed_case_id="mc_456",
            corpus_key="test_corpus",
            test_case_language=LanaguageCode.de,
        )

        debug_template = (
            "https://debug.example.com/{corpus_key}/{case_manager_id}/{managed_case_id}"
        )
        conversation_template = (
            "https://chat.example.com/{corpus_key}/{case_manager_id}/{managed_case_id}"
        )

        debug_url, conversation_url = calc_view_urls(
            test_result, debug_template, conversation_template
        )

        assert debug_url == "https://debug.example.com/test_corpus/cm_123/mc_456"
        assert conversation_url == "https://chat.example.com/test_corpus/cm_123/mc_456"

    def test_calc_view_urls_missing_ids(self):
        """Test view URL calculation with missing IDs"""
        test_result = BotTestResultRecord(
            test_name="Test",
            test_result=BotTestResult(
                test_passed=BotTestResultStatus.OK, description="Test"
            ),
            test_time="2023-01-01T00:00:00Z",
            test_case_id="test_id",
            test_case_description="Test",
            case_manager_id="",  # Missing
            managed_case_id="mc_456",
            corpus_key="test_corpus",
            test_case_language=LanaguageCode.de,
        )

        debug_template = (
            "https://debug.example.com/{corpus_key}/{case_manager_id}/{managed_case_id}"
        )

        debug_url, conversation_url = calc_view_urls(test_result, debug_template)

        assert debug_url is None
        assert conversation_url is None


class TestConversationUtilities:
    """Test conversation utility functions"""

    def test_format_conversation(self):
        """Test conversation formatting"""
        conv = {
            "conversations_list": [
                {
                    "conversation_items": [
                        {"display_role": "USER", "content": "Hello"},
                        {"display_role": "ASSISTANT", "content": "Hi there!"},
                        {"display_role": "USER", "content": "How are you?"},
                    ]
                }
            ]
        }

        result = format_conversation(conv)
        expected = "USER: Hello\nASSISTANT: Hi there!\nUSER: How are you?"
        assert result == expected

    def test_format_conversation_empty(self):
        """Test formatting empty conversation"""
        conv = {"conversations_list": []}
        result = format_conversation(conv)
        assert result == ""

    def test_last_assistant_message(self):
        """Test getting last assistant message"""
        conv = {
            "conversations_list": [
                {
                    "conversation_items": [
                        {"display_role": "USER", "content": "Hello"},
                        {"display_role": "ASSISTANT", "content": "Hi there!"},
                        {"display_role": "USER", "content": "How are you?"},
                        {"display_role": "ASSISTANT", "content": "I'm doing well!"},
                    ]
                }
            ]
        }

        result = last_assistant_message(conv)
        assert result == "I'm doing well!"

    def test_last_assistant_message_none(self):
        """Test getting last assistant message when none exists"""
        conv = {
            "conversations_list": [
                {
                    "conversation_items": [
                        {"display_role": "USER", "content": "Hello"},
                        {"display_role": "USER", "content": "How are you?"},
                    ]
                }
            ]
        }

        result = last_assistant_message(conv)
        assert result is None

    def test_is_completed(self):
        """Test case completion check"""
        conv_completed = {"case_state": "CASE_COMPLETED"}
        conv_waiting = {"case_state": "WAITING_FOR_USER_INPUT"}
        conv_processing = {"case_state": "PROCESSING"}

        assert is_completed(conv_completed) is True
        assert is_completed(conv_waiting) is False
        assert is_completed(conv_processing) is False


class TestApiInteractions:
    """Test API interaction functions (mocked)"""

    @patch("bot_tester.requests.get")
    def test_create_new_case_success(self, mock_get):
        """Test successful case creation"""
        # Mock corpus metadata response
        mock_get.return_value.json.return_value = {
            "ui_settings": {
                "case_manager": {"case_manager_default_case_manager_id": "cm_123"}
            }
        }
        mock_get.return_value.raise_for_status.return_value = None

        # Mock case creation response
        with patch("bot_tester.requests.post") as mock_post:
            mock_post.return_value.json.return_value = {
                "managed_case_id": "mc_456",
                "case_manager_id": "cm_123",
            }
            mock_post.return_value.raise_for_status.return_value = None

            result = create_new_case("https://api.example.com/test_corpus", {}, "test")

            assert result["managed_case_id"] == "mc_456"
            assert result["case_manager_id"] == "cm_123"

    @patch("bot_tester.requests.get")
    def test_get_case_conversation_success(self, mock_get):
        """Test successful conversation retrieval"""
        expected_conversation = {
            "case_state": "WAITING_FOR_USER_INPUT",
            "current_conversation_id": "conv_123",
            "conversations_list": [],
        }
        mock_get.return_value.json.return_value = expected_conversation
        mock_get.return_value.raise_for_status.return_value = None

        case = {"case_manager_id": "cm_123", "managed_case_id": "mc_456"}
        result = get_case_conversation("https://api.example.com/test_corpus", case)

        assert result == expected_conversation

    @patch("bot_tester.requests.post")
    def test_send_user_message_success(self, mock_post):
        """Test successful user message sending"""
        with patch("bot_tester.get_case_conversation") as mock_get_conv:
            mock_get_conv.return_value = {
                "case_state": "WAITING_FOR_USER_INPUT",
                "current_conversation_id": "conv_123",
            }

            mock_post.return_value.json.return_value = {"status": "success"}
            mock_post.return_value.raise_for_status.return_value = None
            mock_post.return_value.status_code = 200

            case = {"case_manager_id": "cm_123", "managed_case_id": "mc_456"}
            result = send_user_message(
                "https://api.example.com/test_corpus",
                case,
                {},
                "test_corpus",
                "Hello",
            )

            assert result == {"status": "success"}

    @patch("bot_tester.requests.post")
    def test_send_user_message_rate_limit_retry(self, mock_post):
        """Test user message sending with rate limit retry"""
        with patch("bot_tester.get_case_conversation") as mock_get_conv:
            mock_get_conv.return_value = {
                "case_state": "WAITING_FOR_USER_INPUT",
                "current_conversation_id": "conv_123",
            }

            # First call returns 429 (rate limit)
            # Second call succeeds
            mock_response_429 = Mock()
            mock_response_429.status_code = 429
            mock_response_success = Mock()
            mock_response_success.status_code = 200
            mock_response_success.json.return_value = {"status": "success"}
            mock_response_success.raise_for_status.return_value = None

            mock_post.side_effect = [mock_response_429, mock_response_success]

            with patch("bot_tester.time.sleep"):  # Mock sleep to speed up test
                case = {"case_manager_id": "cm_123", "managed_case_id": "mc_456"}
                result = send_user_message(
                    "https://api.example.com/test_corpus",
                    case,
                    {},
                    "test_corpus",
                    "Hello",
                )

                assert result == {"status": "success"}
                assert mock_post.call_count == 2

    @patch("bot_tester.get_case_conversation")
    @patch("bot_tester.time.sleep")
    def test_poll_until_user_turn_success(self, mock_sleep, mock_get_conv):
        """Test successful polling until user turn"""
        # First call: bot is still processing
        # Second call: waiting for user input
        mock_get_conv.side_effect = [
            {"case_state": "PROCESSING"},
            {"case_state": "WAITING_FOR_USER_INPUT"},
        ]

        case = {"case_manager_id": "cm_123", "managed_case_id": "mc_456"}
        result = poll_until_user_turn(
            "https://api.example.com/test_corpus", case, {}, test_name="test"
        )

        assert result["case_state"] == "WAITING_FOR_USER_INPUT"
        assert mock_get_conv.call_count == 2

    @patch("bot_tester.get_case_conversation")
    @patch("bot_tester.time.sleep")
    def test_poll_until_user_turn_timeout(self, mock_sleep, mock_get_conv):
        """Test polling timeout"""
        mock_get_conv.return_value = {"case_state": "PROCESSING"}

        case = {"case_manager_id": "cm_123", "managed_case_id": "mc_456"}

        with pytest.raises(RuntimeError, match="Timeout waiting for bot response"):
            poll_until_user_turn(
                "https://api.example.com/test_corpus",
                case,
                {},
                max_retries=2,
                test_name="test",
            )


class TestReportGeneration:
    """Test report generation functionality"""

    def test_calc_report_html_basic(self):
        """Test basic HTML report generation"""
        test_results_report = BotTestResultsReport(
            test_results=[
                BotTestResultRecord(
                    test_name="Test 1",
                    test_result=BotTestResult(
                        test_passed=BotTestResultStatus.OK, description="Passed"
                    ),
                    test_time="2023-01-01T00:00:00Z",
                    test_case_id="test_id",
                    test_case_description="Test description",
                    case_manager_id="cm_123",
                    managed_case_id="mc_456",
                    corpus_key="test_corpus",
                    test_case_language=LanaguageCode.de,
                )
            ],
            test_set_title="Test Suite",
            test_run_start_time="2023-01-01T00:00:00+00:00",
        )

        html = calc_report_html(test_results_report)

        assert "Chatbot Test Report (Test Suite)" in html
        assert "Test 1" in html
        assert "text-success" in html  # Success styling for OK status
        assert "Passed" in html

    def test_calc_report_html_mixed_results(self):
        """Test HTML report with mixed test results"""
        test_results_report = BotTestResultsReport(
            test_results=[
                BotTestResultRecord(
                    test_name="Test OK",
                    test_result=BotTestResult(
                        test_passed=BotTestResultStatus.OK, description="Passed"
                    ),
                    test_time="2023-01-01T00:00:00Z",
                    test_case_id="test_id",
                    test_case_description="Test description",
                    case_manager_id="cm_123",
                    managed_case_id="mc_456",
                    corpus_key="test_corpus",
                    test_case_language=LanaguageCode.de,
                ),
                BotTestResultRecord(
                    test_name="Test NOK",
                    test_result=BotTestResult(
                        test_passed=BotTestResultStatus.NOK, description="Failed"
                    ),
                    test_time="2023-01-01T00:00:00Z",
                    test_case_id="test_id",
                    test_case_description="Test description",
                    case_manager_id="cm_123",
                    managed_case_id="mc_456",
                    corpus_key="test_corpus",
                    test_case_language=LanaguageCode.en,
                ),
                BotTestResultRecord(
                    test_name="Test Skipped",
                    test_result=BotTestResult(
                        test_passed=BotTestResultStatus.SKIPPED, description="Skipped"
                    ),
                    test_time="2023-01-01T00:00:00Z",
                    test_case_id="test_id",
                    test_case_description="Test description",
                    case_manager_id="cm_123",
                    managed_case_id="mc_456",
                    corpus_key="test_corpus",
                    test_case_language=LanaguageCode.fr,
                ),
            ],
            test_set_title="Mixed Results",
            test_run_start_time="2023-01-01T00:00:00+00:00",
        )

        html = calc_report_html(test_results_report)

        assert "Test OK" in html
        assert "Test NOK" in html
        assert "Test Skipped" in html
        assert "text-success" in html  # OK styling
        assert "text-danger" in html  # NOK styling
        assert "text-secondary" in html  # SKIPPED styling

    @patch("bot_tester.boto3.client")
    def test_publish_report_to_s3_success(self, mock_boto_client):
        """Test successful S3 report publishing"""
        mock_s3_client = Mock()
        mock_boto_client.return_value = mock_s3_client

        test_results_report = BotTestResultsReport(
            test_results=[],
            test_set_title="Test",
            test_run_start_time="2023-01-01T00:00:00+00:00",
        )

        with patch("bot_tester.os.environ.get") as mock_env:
            mock_env.return_value = "eu-central-2"

            result_url = publish_report_to_s3(
                "s3://test-bucket/reports/test.html", test_results_report
            )

            # Verify S3 client was called correctly
            mock_s3_client.put_object.assert_called_once()
            call_args = mock_s3_client.put_object.call_args
            assert call_args[1]["Bucket"] == "test-bucket"
            assert call_args[1]["Key"] == "reports/test.html"
            assert call_args[1]["ContentType"] == "text/html"

            # Verify return URL
            assert (
                result_url
                == "https://test-bucket.s3.eu-central-2.amazonaws.com/reports/test.html"
            )

    def test_publish_report_to_s3_invalid_url(self):
        """Test S3 publishing with invalid URL"""
        test_results_report = BotTestResultsReport(
            test_results=[],
            test_set_title="Test",
            test_run_start_time="2023-01-01T00:00:00+00:00",
        )

        with pytest.raises(ValueError, match="Invalid S3 URL"):
            publish_report_to_s3("https://not-s3.com/test", test_results_report)


class TestEdgeCases:
    """Test edge cases and error conditions"""

    def test_empty_scenarios_list(self):
        """Test handling empty scenarios list"""
        config_data = {
            "api": {"base_url": "https://api.example.com"},
            "corpus_key": "test_corpus",
            "prompt_template": "Test template",
            "scenarios": [],  # Empty list
        }
        config = Config(**config_data)
        assert len(config.scenarios) == 0

    def test_scenario_with_unsupported_language(self):
        """Test scenario with invalid language code (should raise ValidationError)"""
        # This should raise a ValidationError since LanaguageCode is a strict enum
        with pytest.raises(ValidationError):
            scenario = Scenario(
                name="Test",
                situation="Test situation",
                scenario_languages=["xx"],  # Non-existent language code
            )

    def test_conversation_with_missing_fields(self):
        """Test conversation handling with missing fields"""
        conv = {"conversations_list": [{"conversation_items": []}]}
        result = format_conversation(conv)
        assert result == ""

    def test_conversation_with_malformed_items(self):
        """Test conversation with malformed items"""
        conv = {
            "conversations_list": [
                {
                    "conversation_items": [
                        {"display_role": "USER"},  # Missing content
                        {"content": "Hello"},  # Missing display_role
                        {"display_role": "ASSISTANT", "content": "Hi"},  # Valid
                    ]
                }
            ]
        }

        # The function should handle missing fields gracefully
        result = format_conversation(conv)
        # Should only include the valid item
        assert "ASSISTANT: Hi" in result


class TestScenarioExecution:
    """Test scenario execution functionality"""

    @patch("bot_tester.create_new_case")
    @patch("bot_tester.poll_until_user_turn")
    @patch("bot_tester.send_user_message")
    @patch("bot_tester.is_completed")
    @patch("bot_tester.last_assistant_message")
    def test_run_scenario_basic_completion(
        self,
        mock_last_msg,
        mock_is_completed,
        mock_send_message,
        mock_poll,
        mock_create_case,
    ):
        """Test basic scenario completion without AI evaluation"""
        # Setup mocks
        mock_create_case.return_value = {
            "case_manager_id": "cm123",
            "managed_case_id": "mc456",
        }
        mock_poll.return_value = {"case_state": "WAITING_FOR_USER_INPUT"}
        mock_send_message.return_value = {}
        mock_is_completed.side_effect = [
            False,
            True,
        ]  # Not completed first, then completed
        mock_last_msg.return_value = "Bot response"

        # Create test scenario
        scenario = Scenario(
            name="Test Scenario",
            situation="Test situation",
            scenario_languages=[LanaguageCode.en],
        )

        # Create test config
        api_config = ApiConfig(base_url="https://test.example.com")
        config = Config(
            api=api_config,
            corpus_key="test_corpus",
            prompt_template="Test: {situation}",
        )

        # Mock Ask object
        ask_mock = MagicMock()

        # Run scenario
        results = run_scenario(scenario, config, ask_mock, "test_corpus")

        # Verify results
        assert len(results) == 1
        result = results[0]
        assert result.test_name == "Test Scenario"
        assert result.test_result.test_passed == BotTestResultStatus.OK
        assert "Case completed successfully" in result.test_result.description
        assert result.case_manager_id == "cm123"
        assert result.managed_case_id == "mc456"
        assert result.corpus_key == "test_corpus"
        assert result.test_case_language == LanaguageCode.en

    @patch("bot_tester.create_new_case")
    @patch("bot_tester.poll_until_user_turn")
    @patch("bot_tester.send_user_message")
    @patch("bot_tester.is_completed")
    @patch("bot_tester.last_assistant_message")
    @patch("bot_tester.format_conversation")
    def test_run_scenario_with_ai_evaluation_success(
        self,
        mock_format_conv,
        mock_last_msg,
        mock_is_completed,
        mock_send_message,
        mock_poll,
        mock_create_case,
    ):
        """Test scenario with AI evaluation that succeeds"""
        # Setup mocks
        mock_create_case.return_value = {
            "case_manager_id": "cm123",
            "managed_case_id": "mc456",
        }
        mock_poll.return_value = {"case_state": "WAITING_FOR_USER_INPUT"}
        mock_send_message.return_value = {}
        mock_is_completed.return_value = False
        mock_last_msg.return_value = "Bot response"
        mock_format_conv.return_value = "USER: Test\nASSISTANT: Bot response"

        # Create scenario with check template
        scenario = Scenario(
            name="AI Test Scenario",
            situation="Test AI evaluation",
            check_template="Check: {situation}\n{conversation}",
            scenario_languages=[LanaguageCode.de],
        )

        api_config = ApiConfig(base_url="https://test.example.com")
        config = Config(
            api=api_config,
            corpus_key="test_corpus",
            prompt_template="Test: {situation}",
        )

        # Mock Ask object to return success
        ask_mock = MagicMock()
        ask_mock.ask.return_value = "SUCCESS: Test passed"

        results = run_scenario(scenario, config, ask_mock, "test_corpus")

        assert len(results) == 1
        result = results[0]
        assert result.test_result.test_passed == BotTestResultStatus.OK
        assert "Test passed after 1 turns" in result.test_result.description
        assert "SUCCESS: Test passed" in result.test_result.description

    @patch("bot_tester.create_new_case")
    @patch("bot_tester.poll_until_user_turn")
    @patch("bot_tester.send_user_message")
    @patch("bot_tester.is_completed")
    @patch("bot_tester.last_assistant_message")
    @patch("bot_tester.format_conversation")
    def test_run_scenario_max_turns_reached(
        self,
        mock_format_conv,
        mock_last_msg,
        mock_is_completed,
        mock_send_message,
        mock_poll,
        mock_create_case,
    ):
        """Test scenario that reaches maximum turns"""
        # Setup mocks
        mock_create_case.return_value = {
            "case_manager_id": "cm123",
            "managed_case_id": "mc456",
        }
        mock_poll.return_value = {"case_state": "WAITING_FOR_USER_INPUT"}
        mock_send_message.return_value = {}
        mock_is_completed.return_value = False  # Never completes
        mock_last_msg.return_value = "Bot response"
        mock_format_conv.return_value = "USER: Test\nASSISTANT: Bot response"

        scenario = Scenario(
            name="Max Turns Test",
            situation="Test max turns",
            max_turns=2,  # Low limit for testing
            scenario_languages=[LanaguageCode.fr],
        )

        api_config = ApiConfig(base_url="https://test.example.com")
        config = Config(
            api=api_config,
            corpus_key="test_corpus",
            prompt_template="Test: {situation}",
        )

        ask_mock = MagicMock()
        results = run_scenario(scenario, config, ask_mock, "test_corpus")

        assert len(results) == 1
        result = results[0]
        assert result.test_result.test_passed == BotTestResultStatus.NOK
        assert (
            "Test incomplete - reached max turns (2)" in result.test_result.description
        )

    def test_run_scenario_skipped(self):
        """Test skipped scenario"""
        scenario = Scenario(
            name="Skipped Test",
            situation="This should be skipped",
            skip=True,
            scenario_languages=[LanaguageCode.es],
        )

        api_config = ApiConfig(base_url="https://test.example.com")
        config = Config(
            api=api_config,
            corpus_key="test_corpus",
            prompt_template="Test: {situation}",
        )

        ask_mock = MagicMock()
        results = run_scenario(scenario, config, ask_mock, "test_corpus")

        assert len(results) == 1
        result = results[0]
        assert result.test_result.test_passed == BotTestResultStatus.SKIPPED
        assert "Test skipped as configured" in result.test_result.description

    @patch("bot_tester.create_new_case")
    def test_run_scenario_with_exception(self, mock_create_case):
        """Test scenario that raises an exception"""
        mock_create_case.side_effect = requests.exceptions.RequestException("API Error")

        scenario = Scenario(
            name="Error Test",
            situation="This will fail",
            scenario_languages=[LanaguageCode.it],
        )

        api_config = ApiConfig(base_url="https://test.example.com")
        config = Config(
            api=api_config,
            corpus_key="test_corpus",
            prompt_template="Test: {situation}",
        )

        ask_mock = MagicMock()
        results = run_scenario(scenario, config, ask_mock, "test_corpus")

        assert len(results) == 1
        result = results[0]
        assert result.test_result.test_passed == BotTestResultStatus.NOK
        assert "Test failed with error: API Error" in result.test_result.description

    def test_run_scenario_multiple_languages(self):
        """Test scenario with multiple languages"""
        scenario = Scenario(
            name="Multi-Language Test",
            situation="Multi-language test",
            scenario_languages=[LanaguageCode.de, LanaguageCode.en, LanaguageCode.fr],
            skip=True,  # Skip to avoid API calls
        )

        api_config = ApiConfig(base_url="https://test.example.com")
        config = Config(
            api=api_config,
            corpus_key="test_corpus",
            prompt_template="Test: {situation}",
        )

        ask_mock = MagicMock()
        results = run_scenario(scenario, config, ask_mock, "test_corpus")

        # Should return one result per language
        assert len(results) == 3
        languages = [result.test_case_language for result in results]
        assert LanaguageCode.de in languages
        assert LanaguageCode.en in languages
        assert LanaguageCode.fr in languages

        # All should be skipped
        for result in results:
            assert result.test_result.test_passed == BotTestResultStatus.SKIPPED


class TestMainFunctionIntegration:
    """Test main function integration"""

    @patch("bot_tester.load_config")
    @patch("bot_tester.Ask")
    @patch("bot_tester.safe_run_scenario")
    @patch("bot_tester.argparse.ArgumentParser.parse_args")
    @patch("bot_tester.os.environ.get")
    def test_main_function_ask_creation(
        self,
        mock_env_get,
        mock_parse_args,
        mock_safe_run,
        mock_ask_class,
        mock_load_config,
        tmp_path,
    ):
        """Test that main function creates Ask object correctly with model from config"""
        # Mock config with model field
        mock_config = Config(
            api=ApiConfig(base_url="https://test.com"),
            corpus_key="test_corpus",
            model="GPT_4_o_CHATGPT_LATEST",
            prompt_template="Test: {situation}",
            scenarios=[],
        )
        mock_load_config.return_value = mock_config

        # Mock args with temporary directory
        mock_args = Mock()
        mock_args.config = "test.yaml"
        mock_args.corpus_key = "test_corpus"
        mock_args.results_dir = str(tmp_path)  # Use pytest tmp_path fixture
        mock_args.publish_s3_url = None
        mock_args.debug_url_template = None
        mock_args.conversation_url_template = None
        mock_parse_args.return_value = mock_args

        # Mock environment
        mock_env_get.return_value = "test_openai_key"

        # Mock Ask instance
        mock_ask_instance = Mock()
        mock_ask_class.return_value = mock_ask_instance

        # Mock safe_run_scenario to return empty results
        mock_safe_run.return_value = []

        # Import and run main
        from bot_tester import main

        main()

        # Verify Ask was created with correct model
        mock_ask_class.assert_called_once()
        call_args = mock_ask_class.call_args
        assert call_args[1]["openai_api_key"] == "test_openai_key"
        # The model should be accessed as an attribute from Ask.OPENAI_MODEL
        assert "model" in call_args[1]

        # Verify that test results file would be created in the temporary directory
        expected_results_file = tmp_path / "test_results.json"
        # The file should exist after running main (mocked scenario returns empty results)
        assert expected_results_file.exists()

    @patch("bot_tester.load_config")
    @patch("bot_tester.Ask")
    @patch("bot_tester.safe_run_scenario")
    @patch("bot_tester.argparse.ArgumentParser.parse_args")
    @patch("bot_tester.os.environ.get")
    def test_main_function_results_file_output(
        self,
        mock_env_get,
        mock_parse_args,
        mock_safe_run,
        mock_ask_class,
        mock_load_config,
        tmp_path,
    ):
        """Test that main function creates results file in specified directory"""
        # Mock config with at least one scenario
        mock_scenario = Scenario(
            name="Test Scenario",
            situation="Test situation"
        )
        mock_config = Config(
            api=ApiConfig(base_url="https://test.com"),
            corpus_key="test_corpus",
            prompt_template="Test: {situation}",
            scenarios=[mock_scenario],  # Add a scenario so safe_run_scenario gets called
        )
        mock_load_config.return_value = mock_config

        # Mock args with temporary directory
        mock_args = Mock()
        mock_args.config = "test.yaml"
        mock_args.corpus_key = "test_corpus"
        mock_args.results_dir = str(tmp_path)
        mock_args.publish_s3_url = None
        mock_args.debug_url_template = None
        mock_args.conversation_url_template = None
        mock_parse_args.return_value = mock_args

        # Mock environment and Ask
        mock_env_get.return_value = "test_openai_key"
        mock_ask_instance = Mock()
        mock_ask_class.return_value = mock_ask_instance

        # Mock safe_run_scenario to return a test result
        mock_test_result = BotTestResultRecord(
            test_name="Test Result",
            test_result=BotTestResult(
                test_passed=BotTestResultStatus.OK, description="Test passed"
            ),
            test_time="2023-01-01T00:00:00Z",
            test_case_id="test_id",
            test_case_description="Test description",
            case_manager_id="cm_123",
            managed_case_id="mc_456",
            corpus_key="test_corpus",
            test_case_language=LanaguageCode.de,
        )
        mock_safe_run.return_value = [mock_test_result]

        # Import and run main
        from bot_tester import main

        main()

        # Verify that test results file was created in the temporary directory
        results_file = tmp_path / "test_results.json"
        assert results_file.exists()

        # Verify the contents of the results file
        import json

        with open(results_file) as f:
            results_data = json.load(f)

        assert "test_results" in results_data
        assert len(results_data["test_results"]) == 1
        assert results_data["test_results"][0]["test_name"] == "Test Result"


class TestSafeRunScenario:
    """Test safe scenario execution wrapper"""

    def test_safe_run_scenario_success(self):
        """Test safe_run_scenario with successful execution"""
        scenario = Scenario(
            name="Safe Test", situation="Safe execution test", skip=True
        )

        api_config = ApiConfig(base_url="https://test.example.com")
        config = Config(
            api=api_config,
            corpus_key="test_corpus",
            prompt_template="Test: {situation}",
        )

        ask_mock = MagicMock()
        results = safe_run_scenario(scenario, config, ask_mock, "test_corpus")

        assert len(results) == 1
        assert results[0].test_result.test_passed == BotTestResultStatus.SKIPPED

    @patch("bot_tester.run_scenario")
    def test_safe_run_scenario_with_exception(self, mock_run_scenario):
        """Test safe_run_scenario when run_scenario raises exception"""
        mock_run_scenario.side_effect = Exception("Unexpected error")

        scenario = Scenario(name="Error Test", situation="This will cause an error")

        api_config = ApiConfig(base_url="https://test.example.com")
        config = Config(
            api=api_config,
            corpus_key="test_corpus",
            prompt_template="Test: {situation}",
        )

        ask_mock = MagicMock()
        results = safe_run_scenario(scenario, config, ask_mock, "test_corpus")

        assert len(results) == 1
        result = results[0]
        assert result.test_result.test_passed == BotTestResultStatus.NOK
        assert (
            "Scenario failed with error: Unexpected error"
            in result.test_result.description
        )
        assert result.test_name == "Error Test"

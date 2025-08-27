# poemai Bot Tester GitHub Action

A comprehensive GitHub Action for automated testing of poemai chatbots with AI-powered evaluation and detailed reporting.

## Features

- **Multi-language Testing**: Test bots in multiple languages with configurable scenarios
- **AI-Powered Evaluation**: Uses OpenAI to intelligently evaluate conversation outcomes
- **Comprehensive Reporting**: Generates detailed HTML reports with conversation links
- **Parallel Execution**: Run multiple test scenarios simultaneously for faster results
- **S3 Integration**: Publish test reports to S3 for easy sharing and archival
- **Flexible Configuration**: YAML-based test scenarios with customizable parameters

## Usage

### Basic Example

```yaml
name: Test Bot
on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  test-bot:
    runs-on: ubuntu-latest
    permissions:
      id-token: write
      contents: read
    
    steps:
      - uses: actions/checkout@v4
      
      - name: Test poemai Bot
        uses: poemAI-ch/poemai-bot-tester-github-action@v1
        with:
          config-file: 'tests/bot-scenarios.yaml'
          corpus-key: 'my-test-corpus'
          role-to-assume: ${{ secrets.AWS_ROLE_ARN }}
          openai-api-key: ${{ secrets.OPENAI_API_KEY }}
          publish-s3-url: 's3://my-reports-bucket/bot-tests/report.html'
```

### Advanced Example with Custom URLs

```yaml
- name: Test poemai Bot
  uses: poemAI-ch/poemai-bot-tester-github-action@v1
  with:
    config-file: 'tests/comprehensive-scenarios.yaml'
    corpus-key: 'staging-corpus'
    role-to-assume: ${{ secrets.AWS_ROLE_ARN }}
    openai-api-key: ${{ secrets.OPENAI_API_KEY }}
    publish-s3-url: 's3://my-reports-bucket/bot-tests/test-${{ github.run_id }}.html'
    debug-url-template: 'https://app.staging.poemai.ch/debug/{corpus_key}/{case_manager_id}/{managed_case_id}'
    conversation-url-template: 'https://app.staging.poemai.ch/conversation/{corpus_key}/{case_manager_id}/{managed_case_id}'
    max-workers: '6'
```

## Inputs

| Input | Required | Default | Description |
|-------|----------|---------|-------------|
| `config-file` | Yes | - | Path to the test scenarios configuration YAML file |
| `corpus-key` | Yes | - | Corpus key to test against |
| `role-to-assume` | Yes | - | ARN of the AWS IAM role to assume |
| `openai-api-key` | Yes | - | OpenAI API key for AI-powered test evaluation |
| `aws-region` | No | `eu-central-2` | AWS region |
| `python-version` | No | `3.12` | Python version to use |
| `publish-s3-url` | No | - | S3 URL to publish the HTML test report |
| `debug-url-template` | No | - | URL template for debug links |
| `conversation-url-template` | No | - | URL template for conversation links |
| `max-workers` | No | `4` | Maximum number of parallel test workers |

## Outputs

| Output | Description |
|--------|-------------|
| `test-results-file` | Path to the generated test results JSON file |
| `report-url` | URL of the published HTML report (if S3 publishing is enabled) |
| `tests-passed` | Number of tests that passed |
| `tests-failed` | Number of tests that failed |
| `tests-skipped` | Number of tests that were skipped |

## Configuration File Format

Create a YAML configuration file defining your test scenarios:

```yaml
# Test configuration
api:
  base_url: "https://api.staging.poemai.ch"

corpus_key: "my-test-corpus"
test_set_title: "My Bot Test Suite"
prompt_template: "You are testing a chatbot. {situation}"

# Global check template for AI evaluation
check_template: |
  Evaluate if this conversation successfully addresses the situation: "{situation}"
  
  Conversation:
  {conversation}
  
  Instructions: {check_instructions}
  
  Respond with SUCCESS if the bot handled the situation appropriately, or FAILURE with explanation.

scenarios:
  - name: "Basic Greeting"
    situation: "Greet the bot and ask for help"
    check_instructions: "Bot should respond politely and offer assistance"
    max_turns: 10
    scenario_languages: ["de", "en"]
    
  - name: "Information Request"
    situation: "Ask for specific information about services"
    check_instructions: "Bot should provide relevant information or guide to resources"
    max_turns: 15
    scenario_languages: ["de"]
    
  - name: "Complex Scenario"
    situation: "Present a complex problem requiring multiple interaction steps"
    check_template: |
      Custom evaluation template for this specific scenario...
      Situation: {situation}
      Conversation: {conversation}
    max_turns: 20
    scenario_languages: ["de", "en", "fr"]
    
  - name: "Skip This Test"
    situation: "This test is not ready yet"
    skip: true
```

### Configuration Options

- **api.base_url**: Base URL for the poemai API
- **corpus_key**: The corpus key to test (can be overridden by input parameter)
- **test_set_title**: Title for the test report
- **prompt_template**: Template for generating conversation prompts
- **check_template**: Global template for AI evaluation (can be overridden per scenario)
- **scenarios**: List of test scenarios

### Scenario Options

- **name**: Unique name for the scenario
- **situation**: Description of the test situation
- **check_instructions**: Instructions for AI evaluation
- **check_template**: Custom evaluation template (overrides global template)
- **max_turns**: Maximum conversation turns (default: 20)
- **scenario_languages**: List of language codes to test (default: ["de"])
- **skip**: Set to true to skip this scenario (default: false)

## Language Support

The action supports testing in multiple languages. Supported language codes include:

- `de` (German)
- `en` (English)
- `fr` (French)
- `es` (Spanish)
- `it` (Italian)
- `pt` (Portuguese)
- `ar` (Arabic)
- `ru` (Russian)
- `zh` (Chinese)
- And many more...

## Report Generation

The action generates comprehensive reports including:

- **JSON Results**: Machine-readable test results with detailed information
- **HTML Report**: Beautiful, shareable report with:
  - Test summary with pass/fail counts
  - Detailed results for each scenario and language
  - Links to debug views and conversation transcripts
  - Bootstrap-styled responsive design

### Sample HTML Report

The generated HTML report includes:
- Test run summary with timestamps
- Sortable table of all test results
- Status indicators (✅ Pass, ❌ Fail, ⏭️ Skip)
- Direct links to conversation debugging tools
- Responsive design for mobile viewing

## AWS Permissions

The assumed IAM role needs the following permissions:

```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Action": [
                "s3:PutObject",
                "s3:PutObjectAcl"
            ],
            "Resource": "arn:aws:s3:::your-reports-bucket/*"
        }
    ]
}
```

## Testing the Action

The action includes comprehensive pytest tests. To run tests locally:

```bash
cd poemai-bot-tester-github-action
pip install -r requirements.txt
pytest tests/ -v
```

## Development

### Project Structure

```
poemai-bot-tester-github-action/
├── action.yaml              # GitHub Action definition
├── bot_tester.py           # Main testing script
├── requirements.txt        # Python dependencies
├── README.md              # This file
└── tests/
    └── test_bot_tester.py # Comprehensive test suite
```

### Key Components

1. **bot_tester.py**: Core testing logic with:
   - Configuration loading and validation
   - Multi-language test execution
   - AI-powered conversation evaluation
   - Report generation (JSON/HTML)
   - S3 publishing integration

2. **action.yaml**: GitHub Action definition with:
   - Input/output specifications
   - AWS credential configuration
   - Python environment setup
   - Test execution and result processing

3. **test_bot_tester.py**: Comprehensive test suite covering:
   - Configuration parsing and validation
   - API interaction mocking
   - Report generation testing
   - Error handling verification

## Error Handling

The action includes robust error handling for:

- **API Failures**: Graceful handling of API errors with retries
- **Rate Limiting**: Automatic backoff for rate-limited requests
- **Configuration Errors**: Clear validation messages for invalid configs
- **Network Issues**: Timeout handling and connection retries
- **AI Service Errors**: Fallback evaluation when OpenAI is unavailable

## Examples

See the `/examples` directory for:
- Sample configuration files
- Integration with different poemai environments
- Complex multi-language testing scenarios
- Custom evaluation templates

## Support

For issues and questions:
- Open an issue in this repository
- Check the action logs for detailed error information
- Review the generated test reports for debugging information

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass with `pytest`
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

# Coverage Analyzer with LLM Integration

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A powerful tool for analyzing functional coverage reports and generating AI-powered test suggestions to accelerate coverage closure in chip verification.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    COVERAGE ANALYZER ARCHITECTURE                           │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   ┌──────────────┐      ┌──────────────┐      ┌──────────────────────┐     │
│   │   Coverage   │      │     LLM      │      │    Prioritization    │     │
│   │    Report    │─────▶│   Analysis   │─────▶│       Engine         │     │
│   │   (Input)    │      │              │      │                      │     │
│   └──────────────┘      └──────────────┘      └──────────────────────┘     │
│          │                     │                        │                   │
│          ▼                     ▼                        ▼                   │
│   ┌──────────────┐      ┌──────────────┐      ┌──────────────────────┐     │
│   │    Parser    │      │   OpenAI /   │      │   Ranked Test        │     │
│   │  (Pydantic)  │      │  Anthropic / │      │   Suggestions        │     │
│   │              │      │   Ollama     │      │                      │     │
│   └──────────────┘      └──────────────┘      └──────────────────────┘     │
│                                │                        │                   │
│                    ┌───────────┴───────────┐            │                   │
│                    │  Cache + Rate Limit   │            ▼                   │
│                    └───────────────────────┘   ┌──────────────────────┐     │
│                                                │  Closure Predictor   │     │
│                                                │  (Time, Probability) │     │
│                                                └──────────────────────┘     │
│                                                          │                   │
│                              ┌───────────────────────────┘                   │
│                              ▼                                               │
│                    ┌───────────────────┐                                    │
│                    │   CLI / Web UI    │                                    │
│                    │    (Output)       │                                    │
│                    └───────────────────┘                                    │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## 🚀 Features

- **📄 Coverage Report Parser**: Robust parsing of functional coverage reports with support for covergroups, coverpoints, bins, and cross-coverage
- **🤖 LLM-Powered Suggestions**: Generate intelligent test scenarios using OpenAI, Anthropic Claude, or local Ollama models
- **📊 Priority Scoring**: Automatic prioritization using the formula: `(Coverage Impact × 0.4) + (Inverse Difficulty × 0.3) + (Dependency Score × 0.3)`
- **🔮 Closure Prediction**: Estimate time-to-closure, closure probability, and identify blocking bins
- **💾 Response Caching**: Reduce API costs with intelligent caching
- **⏱️ Rate Limiting**: Built-in rate limiting to prevent quota exhaustion
- **🖥️ Multiple Interfaces**: CLI tool and Streamlit web UI

## 📋 Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage](#usage)
  - [Command Line Interface](#command-line-interface)
  - [Web Interface](#web-interface)
  - [Python API](#python-api)
- [Configuration](#configuration)
- [Examples](#examples)
- [Architecture](#architecture)
- [Contributing](#contributing)
- [License](#license)

## 🔧 Installation

### Prerequisites

- Python 3.9 or higher
- pip or poetry for package management

### Install from Source

```bash
# Clone the repository
git clone https://github.com/yourusername/coverage-analyzer.git
cd coverage-analyzer

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Or using pip with pyproject.toml
pip install -e .
```

### Quick Install

```bash
pip install -r requirements.txt
```

## ⚡ Quick Start

### 1. Configure API Keys

Copy the example environment file and add your API keys:

```bash
cp .env.example .env
```

Edit `.env` with your preferred editor:

```env
# Choose your LLM provider
LLM_PROVIDER=openai

# OpenAI Configuration
OPENAI_API_KEY=sk-your-api-key-here
OPENAI_MODEL=gpt-4-turbo-preview

# Or Anthropic Configuration
ANTHROPIC_API_KEY=sk-ant-your-api-key-here
ANTHROPIC_MODEL=claude-3-sonnet-20240229

# Or Ollama (local, no API key needed)
OLLAMA_HOST=http://localhost:11434
OLLAMA_MODEL=llama2
```

### 2. Run Demo

```bash
python main.py demo
```

This runs a complete analysis on a sample DMA controller coverage report.

### 3. Analyze Your Coverage Report

```bash
python main.py analyze your_coverage_report.txt -o results.json
```

## 📖 Usage

### Command Line Interface

The CLI provides several commands for different use cases:

#### Full Analysis

```bash
# Basic analysis with LLM suggestions
python main.py analyze coverage_report.txt

# Save results to JSON
python main.py analyze coverage_report.txt --output results.json

# Use a specific LLM provider
python main.py analyze coverage_report.txt --provider openai

# Disable LLM (use mock suggestions)
python main.py analyze coverage_report.txt --no-llm

# Verbose output
python main.py analyze coverage_report.txt --verbose
```

#### Parse Only

```bash
# Parse and display structured output
python main.py parse coverage_report.txt

# Save parsed output to JSON
python main.py parse coverage_report.txt --output parsed.json
```

#### Coverage Prediction

```bash
# Get closure predictions
python main.py predict coverage_report.txt

# With deadline constraint
python main.py predict coverage_report.txt --deadline 24
```

#### Generate Suggestions

```bash
# Get suggestions for all uncovered bins
python main.py suggest coverage_report.txt

# Get suggestions for a specific bin
python main.py suggest coverage_report.txt --bin "cg_error_scenarios.cp_error_type.decode_error"

# Limit number of suggestions
python main.py suggest coverage_report.txt --top 10
```

### Web Interface

Launch the Streamlit web UI:

```bash
streamlit run web_ui.py
```

Then open http://localhost:8501 in your browser.

The web interface provides:
- File upload or paste coverage reports
- Interactive dashboard with coverage metrics
- Filterable suggestions list
- Closure prediction visualization

### Python API

Use the analyzer programmatically in your Python code:

```python
from src.parser.coverage_parser import CoverageParser
from src.analyzer.coverage_analyzer import CoverageAnalyzer
from src.predictor.closure_predictor import ClosurePredictor

# Parse a coverage report
parser = CoverageParser()
report = parser.parse(coverage_text)

# Print summary
print(f"Design: {report.design}")
print(f"Overall Coverage: {report.overall_coverage}%")
print(f"Uncovered Bins: {report.uncovered_count}")

# Full analysis with suggestions
analyzer = CoverageAnalyzer(use_llm=True)
result = analyzer.analyze(coverage_text)

# Print top suggestions
for suggestion in result.suggestions[:5]:
    print(f"{suggestion.rank}. {suggestion.target_bin}")
    print(f"   Score: {suggestion.priority_score:.3f}")
    print(f"   {suggestion.suggestion}")
    print()

# Get closure prediction
predictor = ClosurePredictor(report, result.suggestions)
prediction = predictor.predict()

print(f"Time to 100%: {prediction.estimated_time_to_closure_hours:.1f} hours")
print(f"Closure Probability: {prediction.closure_probability * 100:.1f}%")
```

## ⚙️ Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `LLM_PROVIDER` | LLM provider (openai/anthropic/ollama) | openai |
| `OPENAI_API_KEY` | OpenAI API key | - |
| `OPENAI_MODEL` | OpenAI model name | gpt-4-turbo-preview |
| `ANTHROPIC_API_KEY` | Anthropic API key | - |
| `ANTHROPIC_MODEL` | Anthropic model name | claude-3-sonnet-20240229 |
| `OLLAMA_HOST` | Ollama server URL | http://localhost:11434 |
| `OLLAMA_MODEL` | Ollama model name | llama2 |
| `RATE_LIMIT_RPM` | Requests per minute | 60 |
| `RATE_LIMIT_TPM` | Tokens per minute | 90000 |
| `CACHE_ENABLED` | Enable response caching | true |
| `CACHE_TTL` | Cache time-to-live (seconds) | 3600 |

### Coverage Report Format

The parser supports the following coverage report format:

```
=======================================================
Functional Coverage Report
Design: your_design_name
Date: YYYY-MM-DD
=======================================================

Covergroup: cg_name
  Coverage: XX.XX% (N/M bins)
  
  Coverpoint: cp_name
    bin bin_name[range]      hits: XXXX    covered|UNCOVERED
    ...

-------------------------------------------------------
Cross Coverage: cross_name
  Coverage: XX.XX% (N/M bins)
  
  <value1, value2>          hits: XXXX    covered|UNCOVERED
  ...

=======================================================
Overall Coverage: XX.XX%
=======================================================
```

## 📁 Examples

### Example 1: DMA Controller Analysis

```bash
python main.py analyze examples/sample_coverage_report.txt -o examples/sample_output.json
```

See [examples/sample_output.json](examples/sample_output.json) for the complete output.

### Example 2: PCIe Controller Analysis

```bash
python main.py analyze examples/custom_coverage_report.txt -o examples/custom_output.json
```

See [examples/custom_output.json](examples/custom_output.json) for the complete output.

### Example Output

```json
{
  "suggestions": [
    {
      "target_bin": "cg_channel_arbitration.cp_active_channels.four_channels",
      "priority": "medium",
      "difficulty": "easy",
      "priority_score": 0.665,
      "suggestion": "Enable and start transfers on exactly 4 DMA channels simultaneously",
      "test_outline": [
        "1. Configure channels 0-3 with valid parameters",
        "2. Enable all 4 channels simultaneously",
        "3. Start transfers and monitor"
      ],
      "reasoning": "Since one, two, and three channels work correctly, the multi-channel infrastructure is functional..."
    }
  ]
}
```

## 🏗️ Architecture

```
coverage-analyzer/
├── src/
│   ├── parser/
│   │   └── coverage_parser.py    # Coverage report parsing
│   ├── analyzer/
│   │   ├── coverage_analyzer.py  # Main analysis orchestration
│   │   └── prioritization.py     # Priority scoring algorithm
│   ├── llm/
│   │   ├── llm_client.py         # LLM provider abstraction
│   │   └── prompts.py            # Prompt engineering
│   ├── predictor/
│   │   └── closure_predictor.py  # Closure prediction model
│   └── utils/
│       ├── cache.py              # Response caching
│       └── rate_limiter.py       # API rate limiting
├── examples/
│   ├── sample_coverage_report.txt
│   ├── custom_coverage_report.txt
│   ├── sample_output.json
│   └── custom_output.json
├── docs/
│   └── DESIGN_DOCUMENT.md        # Detailed design documentation
├── tests/
│   ├── test_parser.py
│   ├── test_analyzer.py
│   └── test_prioritization.py
├── main.py                        # CLI entry point
├── web_ui.py                      # Streamlit web interface
├── requirements.txt
├── pyproject.toml
└── README.md
```

## 🧪 Testing

Run the test suite:

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test file
pytest tests/test_parser.py -v
```

## 📚 Documentation

- [Design Document](docs/DESIGN_DOCUMENT.md) - Detailed design decisions and scalability considerations

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- OpenAI and Anthropic for LLM APIs
- The chip verification community for domain expertise
- Streamlit for the web UI framework

---

Made with ❤️ for the verification community

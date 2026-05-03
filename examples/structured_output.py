"""Structured output example — run() with response_model returns a Pydantic instance

Demonstrates:
  1. MovieReview — title, rating, and a free-text summary
  2. WeatherReport — city, temperature, and conditions
  3. CodeReview — list of issues and an overall score

The agent is instructed to emit valid JSON.  agent.run(response_model=MyModel)
calls MyModel.model_validate_json() on the raw string and returns the typed
Pydantic instance.  A note at the bottom shows how to handle ValidationError.
"""

import json
from typing import List, Optional
from pydantic import BaseModel, Field, ValidationError

from cyclops import Agent, AgentConfig

# ---------------------------------------------------------------------------
# Model configuration
# ---------------------------------------------------------------------------

# Default: Ollama (free, local — install https://ollama.ai then `ollama pull qwen3:4b`)
MODEL = "ollama/qwen3:4b"

# Alternatives:
#   MODEL = "gpt-4o-mini"                      # OPENAI_API_KEY  (better JSON reliability)
#   MODEL = "groq/llama-3.1-8b-instant"        # GROQ_API_KEY    (free, fast)
#   MODEL = "claude-3-haiku-20240307"          # ANTHROPIC_API_KEY


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------

class MovieReview(BaseModel):
    """Structured review of a film."""
    title: str = Field(description="Exact movie title")
    year: int = Field(description="Release year")
    rating: float = Field(ge=0, le=10, description="Rating from 0 to 10")
    summary: str = Field(description="One-paragraph review summary")
    recommended: bool = Field(description="Whether you recommend watching it")


class WeatherReport(BaseModel):
    """Current weather snapshot for a city."""
    city: str
    country_code: str = Field(description="ISO 3166-1 alpha-2 country code, e.g. US")
    temperature_celsius: float
    conditions: str = Field(description="Human-readable description, e.g. Partly cloudy")
    humidity_percent: Optional[int] = Field(default=None, ge=0, le=100)
    wind_kph: Optional[float] = Field(default=None, ge=0)


class CodeIssue(BaseModel):
    """A single issue found during code review."""
    line: Optional[int] = Field(default=None, description="Line number if applicable")
    severity: str = Field(description="One of: critical, warning, info")
    description: str


class CodeReview(BaseModel):
    """Automated code review result."""
    language: str
    score: int = Field(ge=0, le=100, description="Overall quality score 0-100")
    issues: List[CodeIssue] = Field(default_factory=list)
    summary: str = Field(description="Brief overall assessment")
    approved: bool = Field(description="True if code can be merged as-is")


# ---------------------------------------------------------------------------
# Helper: build an agent that always responds in JSON
# ---------------------------------------------------------------------------

def make_agent(schema: type) -> Agent:
    """Return an agent primed to output JSON matching *schema*."""
    schema_json = json.dumps(schema.model_json_schema(), indent=2)
    system_prompt = (
        "You are a helpful assistant that always responds with valid JSON. "
        "Do not include any prose outside the JSON object. "
        f"Your response must conform to this JSON schema:\n{schema_json}"
    )
    config = AgentConfig(
        model=MODEL,
        system_prompt=system_prompt,
        temperature=0.1,  # low temperature for deterministic structured output
    )
    return Agent(config)


# ---------------------------------------------------------------------------
# Demo 1 — Movie review
# ---------------------------------------------------------------------------

def demo_movie_review() -> None:
    print("=" * 60)
    print("1. MovieReview structured output")
    print("=" * 60)

    agent = make_agent(MovieReview)

    review: MovieReview = agent.run(
        "Write a review of the movie Inception (2010).",
        response_model=MovieReview,
    )

    print(f"Title       : {review.title} ({review.year})")
    print(f"Rating      : {review.rating}/10")
    print(f"Recommended : {review.recommended}")
    print(f"Summary     : {review.summary}")
    print()


# ---------------------------------------------------------------------------
# Demo 2 — Weather report
# ---------------------------------------------------------------------------

def demo_weather_report() -> None:
    print("=" * 60)
    print("2. WeatherReport structured output")
    print("=" * 60)

    agent = make_agent(WeatherReport)

    report: WeatherReport = agent.run(
        "Generate a realistic weather report for Tokyo right now.",
        response_model=WeatherReport,
    )

    print(f"City        : {report.city}, {report.country_code}")
    print(f"Temperature : {report.temperature_celsius}°C")
    print(f"Conditions  : {report.conditions}")
    if report.humidity_percent is not None:
        print(f"Humidity    : {report.humidity_percent}%")
    if report.wind_kph is not None:
        print(f"Wind        : {report.wind_kph} kph")
    print()


# ---------------------------------------------------------------------------
# Demo 3 — Code review
# ---------------------------------------------------------------------------

SAMPLE_CODE = """
def divide(a, b):
    return a / b

def fetch_user(user_id):
    import requests
    url = "http://api.example.com/users/" + user_id
    r = requests.get(url)
    return r.json()

PASSWORD = "hunter2"
"""

def demo_code_review() -> None:
    print("=" * 60)
    print("3. CodeReview structured output")
    print("=" * 60)
    print("Code under review:")
    print(SAMPLE_CODE)

    agent = make_agent(CodeReview)

    review: CodeReview = agent.run(
        f"Review the following Python code and identify any issues:\n```python\n{SAMPLE_CODE}\n```",
        response_model=CodeReview,
    )

    print(f"Language : {review.language}")
    print(f"Score    : {review.score}/100")
    print(f"Approved : {review.approved}")
    print(f"Summary  : {review.summary}")
    if review.issues:
        print("\nIssues found:")
        for issue in review.issues:
            loc = f"line {issue.line}" if issue.line else "general"
            print(f"  [{issue.severity.upper():8s}] ({loc}) {issue.description}")
    print()


# ---------------------------------------------------------------------------
# Demo 4 — Handling ValidationError
# ---------------------------------------------------------------------------

def demo_validation_error_handling() -> None:
    print("=" * 60)
    print("4. Handling ValidationError gracefully")
    print("=" * 60)

    # Use a low max_tokens to provoke a truncated/invalid JSON response
    config = AgentConfig(
        model=MODEL,
        system_prompt="Respond only with valid JSON matching the MovieReview schema.",
        temperature=0.1,
        max_tokens=10,   # intentionally too small to trigger a bad response
    )
    agent = Agent(config)

    try:
        review: MovieReview = agent.run(
            "Review Blade Runner 2049.",
            response_model=MovieReview,
        )
        print(f"Parsed successfully: {review.title}")
    except ValidationError as exc:
        # ValidationError is raised by Pydantic if the JSON is malformed or
        # fields do not pass validation (e.g. rating out of range).
        print("Caught ValidationError — the model returned invalid JSON.")
        print("Raw errors:")
        for error in exc.errors():
            print(f"  field={error['loc']}, msg={error['msg']}")
        print("\nHandling strategies:")
        print("  - Retry the request (agent.run() again)")
        print("  - Fall back to raw string: agent.run(prompt)  # no response_model")
        print("  - Use a more capable or instruction-following model")
    except Exception as exc:
        print(f"Other error (e.g. model not running): {exc}")
    print()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    demo_movie_review()
    demo_weather_report()
    demo_code_review()
    demo_validation_error_handling()

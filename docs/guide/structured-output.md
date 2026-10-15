# Structured Output

## response_model parameter

Pass any Pydantic `BaseModel` class as `response_model` to `run()` or `arun()`. Instead of returning a plain string, the agent parses the reply into the model and returns a typed instance. Where the provider supports it, the model's JSON schema is sent with the request (see [How it works](#how-it-works)).

```python
from pydantic import BaseModel
from cyclops import Agent, AgentConfig


class WeatherReport(BaseModel):
    city: str
    temperature_f: float
    condition: str
    humidity_percent: int


config = AgentConfig(model="groq/llama-3.1-8b-instant")
agent = Agent(config)

report = agent.run(
    "Give me a weather report for London right now.",
    response_model=WeatherReport,
)

print(report.city)             # "London"
print(report.temperature_f)   # 61.5
print(report.condition)       # "Overcast"
```

The return type is the Pydantic model instance, not a string. Your IDE will see the correct type.

## How it works

When the agent has no tools and LiteLLM reports that the model supports response schemas (`litellm.supports_response_schema`), Cyclops sends the model's JSON schema as `response_format`, so the provider constrains the reply to it. Through a `router`, every deployment of the model has to support it.

Agents with tools send the schema only when LiteLLM's model map lists native structured output (`supports_native_structured_output`) in the entry for that provider and model, again for every deployment. A model without an entry gets no schema. LiteLLM implements the schema for other models (such as `claude-sonnet-4-20250514`, or `vertex_ai/claude-sonnet-4-5`) as a forced tool call, which would keep the model from calling the agent's tools. Anthropic models from the 4.5 generation on are listed directly and through Bedrock. Providers that always take the schema natively, such as OpenAI and Gemini, are not listed, and neither are most Azure AI Anthropic models, so tool agents on those models get no schema.

Nothing is added to the prompt. Without a schema (a model without `supports_response_schema`, or a tool agent whose model lacks native structured output), the reply is only validated, so ask for JSON matching the model in the system prompt.

Either way the run validates each answer with `response_model.model_validate_json(content)` before it ends, and sends an answer that fails back to the model with the errors (see [Retries](#retries)).

## Examples with various model shapes

**Nested models:**

```python
from typing import List
from pydantic import BaseModel


class Ingredient(BaseModel):
    name: str
    amount: str


class Recipe(BaseModel):
    title: str
    servings: int
    ingredients: List[Ingredient]
    instructions: List[str]


recipe = agent.run(
    "Give me a simple pasta carbonara recipe.",
    response_model=Recipe,
)

for ingredient in recipe.ingredients:
    print(f"  {ingredient.amount} of {ingredient.name}")
```

**Optional fields:**

```python
from typing import Optional
from pydantic import BaseModel


class PersonInfo(BaseModel):
    name: str
    age: Optional[int] = None
    occupation: Optional[str] = None
    fun_fact: str


info = agent.run(
    "Tell me about Albert Einstein.",
    response_model=PersonInfo,
)
print(info.name, info.age)
```

**Boolean and numeric fields:**

```python
from pydantic import BaseModel


class SentimentResult(BaseModel):
    text: str
    sentiment: str          # "positive", "negative", or "neutral"
    confidence: float       # 0.0 to 1.0
    contains_question: bool


result = agent.run(
    'Analyse the sentiment of: "I absolutely love this product!"',
    response_model=SentimentResult,
)
print(result.sentiment, result.confidence)
```

## Async structured output

`arun()` accepts `response_model` too:

```python
import asyncio


async def main():
    result = await agent.arun(
        "List three programming languages and their main use cases.",
        response_model=SomeModel,
    )
    print(result)


asyncio.run(main())
```

## Retries

An answer that fails validation is kept in the conversation and followed by a user message with the validation errors, the way [guard](hooks.md#end-of-run-guards) feedback is, and the model answers again:

````text
1 validation error:
```json
[
  {
    "type": "float_parsing",
    "loc": [
      "temperature_f"
    ],
    "msg": "Input should be a valid number, unable to parse string as a number"
  }
]
```

Fix the errors and try again.
````

`AgentConfig.output_retries` (default `1`) caps how many times per run that happens. If the answer still fails after that, the run ends with `stop_reason="output_validation"` and the last answer as `content`, and `run()` and `arun()` raise its `pydantic.ValidationError`. With `output_retries=0` the first invalid answer ends the run.

```python
config = AgentConfig(model="groq/llama-3.1-8b-instant", output_retries=3)
```

Each retry is a step, so it counts against `max_steps` and the token and cost budgets. When no step or budget is left for a retry, the invalid answer ends the run with `stop_reason="output_validation"` as it does once the retries run out, and the errors are not added to the conversation. Guards judge only an answer that validates. In a streamed run the invalid answer has already streamed when an `OutputRejected` [event](streaming.md) marks it, and the retry streams after it.

## Error handling

When the retries run out, `run()` and `arun()` raise the last answer's `pydantic.ValidationError`. Catch it to fall back to plain text:

```python
from pydantic import ValidationError

try:
    result = agent.run("...", response_model=MyModel)
except ValidationError as e:
    print("Model returned invalid JSON:", e)
    # Optionally fall back:
    text = agent.run("...")
```

`agent.start(..., response_model=MyModel)` raises nothing: the result has `stop_reason="output_validation"` and the last answer as `content`.

/// note
Not all models reliably produce valid JSON on the first attempt. This matters most where no schema is sent. For smaller models, add `"Respond with valid JSON only."` to the system prompt.
///

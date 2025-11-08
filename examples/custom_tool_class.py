"""Custom tool class example"""

from cyclops import Agent, AgentConfig
from cyclops.toolkit.tool import BaseTool
from typing import Any, Optional


class DatabaseTool(BaseTool):
    """Custom tool that maintains state"""

    def __init__(self):
        super().__init__(
            name="query_db", description="Query a simple in-memory database"
        )
        self.database = {
            "users": [
                {"id": 1, "name": "Alice", "age": 30},
                {"id": 2, "name": "Bob", "age": 25},
                {"id": 3, "name": "Charlie", "age": 35},
            ]
        }

    async def execute(
        self, table: str, filter_key: Optional[str] = None, filter_value: Any = None
    ) -> str:
        """Query the database"""
        if table not in self.database:
            return f"Table '{table}' not found"

        data = self.database[table]

        if filter_key and filter_value:
            data = [row for row in data if row.get(filter_key) == filter_value]

        return str(data)


class CalculatorTool(BaseTool):
    """Calculator with history"""

    def __init__(self):
        super().__init__(
            name="calculator", description="Perform calculations with history"
        )
        self.history = []

    async def execute(self, expression: str) -> str:
        """Evaluate a mathematical expression"""
        try:
            result = eval(expression)
            self.history.append(f"{expression} = {result}")
            return f"Result: {result}"
        except Exception as e:
            return f"Error: {str(e)}"


# Create custom tools
db_tool = DatabaseTool()
calc_tool = CalculatorTool()

# Create agent with custom tools
config = AgentConfig(model="ollama/qwen3:4b")
agent = Agent(config, tools=[db_tool, calc_tool])

print("Agent with custom tool classes\n")

# Use database tool
response = agent.run("Show me all users in the database")
print(f"Q: Show me all users in the database\nA: {response}\n")

# Use calculator tool
response = agent.run("Calculate 15 * 7 + 3")
print(f"Q: Calculate 15 * 7 + 3\nA: {response}\n")

# Tool maintains state
print("Calculator history:", calc_tool.history)

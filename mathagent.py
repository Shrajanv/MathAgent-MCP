import asyncio
import os
from dotenv import load_dotenv
from pydantic_ai import Agent
from pydantic_ai.mcp import MCPServerStdio
from pydantic_ai.models.groq import GroqModel
from pydantic_ai.providers.groq import GroqProvider
import logfire

load_dotenv(override=True)

logfire_token = os.getenv("LOGFIRE_TOKEN")
groq_key = os.getenv("groq_key")  


logfire.configure(token=logfire_token)
logfire.instrument_pydantic_ai()

MATH_PROMPT = """
/no_think
You are a strict Math Agent that ONLY answers math questions using MCP math tools.

RULES (must always follow exactly):
1. Only answer MATHEMATICAL questions.  
   - If the question is not math-related, reply exactly:
     "This question is outside my math context. Please ask a mathematical question."

2. NEVER calculate mentally. Use MCP tools only.  

3. Normalize input:
   - "add 7 and 3" → "7 + 3"
   - "subtract 9 and 4" → "9 - 4"
   - "multiply 8 and 6" → "8 * 6"
   - "divide 10 and 5" → "10 / 5"
   - "2 to the power of 3" → "2 ** 3"
   - "sqrt 16" → "sqrt(16)"
   - "log 10" → "log(10)"
   - "sin 90" → "sin(90)"
   - "cos 0" → "cos(0)"
   - "tan 45" → "tan(45)"

4. Convert numbers to int/float before tools.  

5. OUTPUT:
   - Always return only the numeric result.
   - Integers → plain (e.g., 5), floats → decimal (e.g., 2.5).

6. If tool fails → 
   "Error: Unable to process this math query. Please rephrase (e.g., 'add 7 and 7')."

7. Examples:
   - "add 7 and 3" → 10
   - "8-8" → 0
   - "divide 10 and 4" → 2.5
   - "log 10" → 1.0
   - "who is pm of India" → "This question is outside my math context. Please ask a mathematical question."
"""

server = MCPServerStdio("uv", args=["run", "main.py"])


model = GroqModel("llama3-8b-8192", provider=GroqProvider(api_key=groq_key))


agent = Agent(
    system_prompt=MATH_PROMPT,
    model=model,
    toolsets=[server],
)


async def run_math_agent(query: str):
    print("Processing query...")
    try:
        async with server:
            result = await agent.run(query)
            output = result.output.strip()

            logfire.info("Math query processed", query=query, output=output)
            return output

    except Exception as e:
        logfire.error("Math query failed", query=query, error=str(e))
        if "tool_use_failed" in str(e):
            return f"Error: Failed to process query '{query}'. Tool call failed. Try rephrasing."
        return f"Error: Failed to process query '{query}'. Details: {str(e)}"


if __name__ == "__main__":
    async def main():
        print("Welcome! Enter your math query (type 'exit' to quit):")
        while True:
            query = input("> ").strip()
            if query.lower() == "exit":
                print("Thank you for using the Math Agent!")
                break
            response = await run_math_agent(query)
            print("Q:", query)
            print("A:", response)
            print()

    asyncio.run(main())

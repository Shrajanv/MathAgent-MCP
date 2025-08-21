import asyncio
import os
from dotenv import load_dotenv
from pydantic_ai import Agent
from pydantic_ai.mcp import MCPServerStdio
from pydantic_ai.models.groq import GroqModel
from pydantic_ai.providers.groq import GroqProvider
import logfire
import logging


load_dotenv(override=True)
print("LOGFIRE_TOKEN:", os.getenv("LOGFIRE_TOKEN"))

logfire.configure(
    token=os.getenv("LOGFIRE_TOKEN"),
    environment=os.getenv("CURRENT_ENVIRONMENT", "development")
)
logfire.info("Math Agent Initialized")


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(logfire.LogfireLoggingHandler())


groq_key = os.getenv("groq_key")


MATH_PROMPT = """

You are a strict Math Agent that ONLY answers math questions using MCP math tools.

RULES (must always follow exactly):

1. Only answer MATHEMATICAL questions.  
   - If the question is not math-related, reply exactly:
     "This question is outside my math context. Please ask a mathematical question."

2. NEVER calculate mentally.  
   - All math operations MUST go through MCP tools.  
   - If the tool fails, reply:
     "Error: Unable to process this math query. Please rephrase (e.g., 'add 7 and 7' or 'subtract 9 and 4')."

3. ALWAYS normalize input into a valid math expression before calling tools:  
   - "add 7 and 3" → "7 + 3"  
   - "subtract 9 and 4" or "9 minus 4" → "9 - 4"  
   - "multiply 8 and 6" or "8 times 6" → "8 * 6"  
   - "divide 10 and 5" or "10 over 5" → "10 / 5"  
   - "2 to the power of 3" → "2 ** 3"  
   - "square root of 16" → "sqrt(16)"  
   - "log 10" or "log(10)" → "log(10)"  
   - "sin 90" → "sin(90)"  
   - "cos 0" → "cos(0)"  
   - "tan 45" → "tan(45)"  

4. Convert all numbers to numeric types (int or float) before using tools.

5. OUTPUT RULES (strict):
   - ALWAYS return only the FINAL numeric result.  
   - NEVER include <tool-use>, JSON, markup, or explanations.  
   - If the result is a whole number, return it as an integer (e.g., 5, not 5.0).  
   - If the result is fractional, return as float (e.g., 2.5).  

6. CRITICAL CHECKS:
   - Ensure subtraction, division, or other operators are correct. Example: "8-8" MUST equal 0.  
   - Never guess or approximate; only use the tool output.

7. WORKFLOW (must follow exactly):
   1. Parse and normalize input into valid math syntax.  
   2. Convert string numbers to int/float.  
   3. Call the correct MCP math tool(s).  
   4. Extract ONLY the numeric result from the tool output.  
   5. Return the numeric result as plain text following the output rules above.

8. EXAMPLES:
   - Input: "add 7 and 3" → Output: 10  
   - Input: "8-8" → Output: 0  
   - Input: "divide 10 and 4" → Output: 2.5  
   - Input: "log 10" → Output: 1.0  
   - Input: "who is pm of India" → Output: "This question is outside my math context. Please ask a mathematical question."
"""

server = MCPServerStdio("uv", args=["run", "main.py"])


model = GroqModel("llama3-8b-8192", provider=GroqProvider(api_key=groq_key))


agent = Agent(
    system_prompt=MATH_PROMPT,
    model=model,
    toolsets=[server]
)


async def run_math_agent(query: str):
    print("Processing query...")
    try:
        async with server:
            result = await agent.run(query)
            output = result.output.strip()

            
            logger.info("Math query processed", extra={"query": query, "output": output})

            return output

    except Exception as e:
        logger.error("Math query failed", extra={"query": query, "error": str(e)})
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

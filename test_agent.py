import asyncio
import json
import os

from openai import AsyncOpenAI
from openreward import AsyncOpenReward

MODEL_NAME = os.environ.get("MODEL_NAME", "gpt-5.2")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY")

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable must be set")
if not TAVILY_API_KEY:
    raise ValueError("TAVILY_API_KEY environment variable must be set")


async def main() -> None:
    or_client = AsyncOpenReward()
    oai_client = AsyncOpenAI(api_key=OPENAI_API_KEY)

    environment = or_client.environments.get(
        name="GeneralReasoning/PaperSearchQA"
    )
    tasks = await environment.list_tasks(split="train")
    tools = await environment.list_tools(format="openai")

    print(f"Found {len(tasks)} test tasks")

    # Test first task
    task = tasks[0]
    finished = False

    async with environment.session(
        task=task,
        secrets={
            "openai_api_key": OPENAI_API_KEY,
            "tavily_api_key": TAVILY_API_KEY
        }
    ) as session:
        prompt = await session.get_prompt()
        prompt_text = prompt[0].text

        input_list = [{"role": "user", "content": prompt_text}]
        print(f"\n=== PROMPT ===\n{prompt_text}\n")

        turn_count = 0
        max_turns = 10

        while not finished and turn_count < max_turns:
            turn_count += 1
            print(f"\n=== TURN {turn_count} ===")

            response = await oai_client.responses.create(
                model=MODEL_NAME,
                tools=tools,
                input=input_list,
            )

            input_list += response.output
            print(input_list[-1])

            for item in response.output:
                if item.type == "function_call":
                    print(f"\nTool call: {item.name}")
                    print(f"Arguments: {item.arguments}")

                    tool_result = await session.call_tool(
                        item.name,
                        json.loads(str(item.arguments)),
                    )

                    finished = tool_result.finished
                    reward = tool_result.reward
                    result_text = tool_result.blocks[0].text if tool_result.blocks else ""

                    input_list.append({
                        "type": "function_call_output",
                        "call_id": item.call_id,
                        "output": result_text
                    })
                    print(input_list[-1])

                    print(f"\n=== RESULT ===\n{result_text}")
                    print(f"\nReward: {reward:.2f}")

                    if finished:
                        print("\n=== EPISODE FINISHED ===")
                        break

            if not any(i.type == "function_call" for i in response.output):
                print("No tool call, ending.")
                break


if __name__ == "__main__":
    asyncio.run(main())

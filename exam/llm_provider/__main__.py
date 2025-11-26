import asyncio

from exam.llm_provider import get_llm
from autogen_core.models import UserMessage


async def main():

    default_llm = get_llm()
    print(f"\nOttenuto LLM (default):")
    print(f"  Modello: {default_llm}")
    print(f"  Classe: {type(default_llm)}")
    result = await default_llm.create([UserMessage(content="What is the capital of France?", source="user")])
    print(result)
    await default_llm.close()



if __name__ == "__main__":
    asyncio.run(main())

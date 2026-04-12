import asyncio
import sys
from client import FinanceOptimizerEnv

async def main():
    env = FinanceOptimizerEnv("http://localhost:9999")
    try:
        await env.close()
        print("Safely closed")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    asyncio.run(main())

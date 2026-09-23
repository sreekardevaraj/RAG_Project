import asyncio
import sys

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


class RAGMCPClient:

    def __init__(self):

        self.server_params = StdioServerParameters(
            command=sys.executable,
            args=[
                "-m",
                "backend.mcp_server",
            ],
            cwd=".",
        )

        self.stdio_context = None
        self.session = None

    # --------------------------------------------------
    # Connect to MCP Server
    # --------------------------------------------------

    async def connect(self):

        self.stdio_context = stdio_client(
            self.server_params
        )

        read, write = await self.stdio_context.__aenter__()

        self.session = ClientSession(
            read,
            write
        )

        await self.session.__aenter__()

        await self.session.initialize()

        print("MCP server connected successfully.")

    # --------------------------------------------------
    # Initialize Pipeline
    # --------------------------------------------------

    async def initialize_pipeline(
        self,
        force_rebuild: bool = False
    ):

        if self.session is None:
            await self.connect()

        result = await self.session.call_tool(
            "initialize_pipeline",
            {
                "force_rebuild": force_rebuild
            }
        )

        return self._extract_result(result)

    # --------------------------------------------------
    # Ask Agent
    # --------------------------------------------------

    async def ask_agent(
        self,
        question: str
    ):

        if self.session is None:
            await self.connect()

        result = await self.session.call_tool(
            "ask_agent",
            {
                "question": question
            }
        )

        return self._extract_result(result)

    # --------------------------------------------------
    # Pipeline Status
    # --------------------------------------------------

    async def get_pipeline_status(self):

        if self.session is None:
            await self.connect()

        result = await self.session.read_resource(
            "status://pipeline"
        )

        return self._extract_resource_result(result)

    # --------------------------------------------------
    # Server Information
    # --------------------------------------------------

    async def get_server_info(self):

        if self.session is None:
            await self.connect()

        result = await self.session.read_resource(
            "info://server"
        )

        return self._extract_resource_result(result)

    # --------------------------------------------------
    # Extract Tool Result
    # --------------------------------------------------

    @staticmethod
    def _extract_result(result):

        if not result.content:
            return None

        for item in result.content:

            if hasattr(item, "text"):
                return item.text

        return None

    # --------------------------------------------------
    # Extract Resource Result
    # --------------------------------------------------

    @staticmethod
    def _extract_resource_result(result):

        if not result.contents:
            return None

        for item in result.contents:

            if hasattr(item, "text"):
                return item.text

        return None

    # --------------------------------------------------
    # Disconnect
    # --------------------------------------------------

    async def disconnect(self):

        if self.session is not None:

            await self.session.__aexit__(
                None,
                None,
                None
            )

            self.session = None

        if self.stdio_context is not None:

            await self.stdio_context.__aexit__(
                None,
                None,
                None
            )

            self.stdio_context = None


# ======================================================
# TEST
# ======================================================

async def main():

    client = RAGMCPClient()

    try:

        print("\nConnecting to MCP server...")

        await client.connect()

        print("\n--- Pipeline Status ---")

        status = await client.get_pipeline_status()

        print(status)

        print("\n--- Initialize Pipeline ---")

        result = await client.initialize_pipeline()

        print(result)

        print("\n--- Pipeline Status ---")

        status = await client.get_pipeline_status()

        print(status)

        print("\n--- Ask Agent ---")

        result = await client.ask_agent(
            "What is the architecture of this system?"
        )

        print(result)

        print("\n--- Server Information ---")

        info = await client.get_server_info()

        print(info)

    finally:

        await client.disconnect()


if __name__ == "__main__":

    asyncio.run(main())

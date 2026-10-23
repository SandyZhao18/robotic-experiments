import os

from dotenv import load_dotenv
from viam.robot.client import RobotClient


async def connect() -> RobotClient:
    """Connect to the machine using credentials from the environment (or a .env file)."""
    load_dotenv()
    opts = RobotClient.Options.with_api_key(
        api_key=os.environ["VIAM_API_KEY"],
        api_key_id=os.environ["VIAM_API_KEY_ID"],
    )
    return await RobotClient.at_address(os.environ["VIAM_MACHINE_ADDRESS"], opts)

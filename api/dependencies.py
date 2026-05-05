from fastapi import Header, HTTPException
from config import AI_SECRET_KEY


async def verify_api_key(x_api_key: str = Header(...)):
    """
    Checks X-Api-Key header on every request.
    React sends this header via aiClient.js
    """
    if x_api_key != AI_SECRET_KEY:
        raise HTTPException(
            status_code=403,
            detail="Invalid API key."
        )
import os

from evaluation_end_points.settings import get_settings

from evaluation_end_points.evaluate.router import router

from evaluation_end_points.utils.app import app


settings = get_settings()
app.include_router(router)


@app.get("/ping", status_code=200)
async def get_ping():
    return 200

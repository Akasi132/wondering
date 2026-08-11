"""Serve the site. This is the entrypoint a host runs to bring the app up.

    python server.py

Binds to $HOST:$PORT so it works unchanged wherever those are injected. Wasmer Edge sets
PORT; the equivalent command form, if a host wants one instead of a script, is

    uvicorn app.api:app --host 0.0.0.0 --port $PORT

Both start the same object. For local development, prefer `uvicorn app.api:app --reload`.
"""

import os

import uvicorn

from app.api import app


def main() -> None:
    # 0.0.0.0 rather than 127.0.0.1: inside a container or a wasm instance, a loopback bind
    # is unreachable from outside and the platform's health check fails with no useful error.
    uvicorn.run(
        app,
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", "8000")),
        # The generation call can run for minutes; the default keep-alive would drop a
        # waiting browser before the path is ready.
        timeout_keep_alive=300,
    )


if __name__ == "__main__":
    main()

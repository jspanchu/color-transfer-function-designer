import asyncio
import logging

from trame.app import get_server

from color_transfer_function_designer.app.core import App
from color_transfer_function_designer.app.logger import install_handlers

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
install_handlers(logger)


async def real_main(server=None):
    if server is None:
        server = get_server(server)

    assert server is not None
    server.cli.add_argument(
        "--data-directory",
        help="Path to show in the file browser by default",
        default=None,
        type=str,
    )
    server.cli.add_argument(  # pyright: ignore[reportOptionalMemberAccess]
        "--config",
        help="Path to a JSON file with default ref_file, tf_file, tgt_file paths",
        default=None,
        type=str,
    )
    args = server.cli.parse_args()  # pyright: ignore[reportOptionalMemberAccess]
    logger.debug(f"Program {args=}")  # noqa: G004
    app = App(
        args.data_directory,
        config=args.config,
        server=server,
    )
    await app.server.start(exec_mode="coroutine")


def main():
    asyncio.run(real_main())


if __name__ == "__main__":
    main()

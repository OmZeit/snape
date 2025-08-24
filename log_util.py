import logging


def setup_logging(log_file: str = "app.log"):
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(),
        ],
    )
    logger = logging.getLogger("log_util")
    logger.info("Logging is set up.")
    return logger


logger = setup_logging()


def log_request(request):
    """
    Log details of an incoming Flask request.
    """
    logger.info(f"Received request: {request.method} {request.url}")
    logger.debug(f"Headers: {request.headers}")
    logger.debug(f"Body: {request.get_data(as_text=True)}")
    logger.debug(f"Remote address: {request.remote_addr}")
    logger.debug(f"User agent: {request.headers.get('User-Agent')}")


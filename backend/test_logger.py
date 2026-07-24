from app.core.logger import get_logger

logger = get_logger(__name__)

logger.debug("Debug Log")

logger.info("Information Log")

logger.warning("Warning Log")

logger.error("Error Log")

logger.critical("Critical Log")
import logging
import yaml

# Ref - https://medium.com/pythoneers/beyond-print-statements-elevating-debugging-with-python-logging-715b2ae36cd5

def logger_helper():

    with open("config.yaml") as f:
      cfg = yaml.load(f, Loader=yaml.FullLoader)

    logger = logging.getLogger('my_logger')
    logger.setLevel(logging.DEBUG)  # Capture all messages of debug or higher severity

    ### File handler for errors
    # Create a file handler that writes log messages to 'error.log'
    file_handler = logging.FileHandler('error.log') 
    # Set the logging level for this handler to ERROR, which means it will only handle messages of ERROR level or higher
    file_handler.setLevel(logging.ERROR)  

    ### Console handler for info and above
    # Create a console handler that writes log messages to the console
    console_handler = logging.StreamHandler()  
    
    if cfg['debug'] == True:
        console_handler.setLevel(logging.DEBUG)  
    else:
        # Set the logging level for this handler to INFO, which means it will handle messages of INFO level or higher
        console_handler.setLevel(logging.INFO)  

    ### Set formats for handlers
    # Define the format of log messages
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s') 
    # Apply the formatter to the file handler 
    file_handler.setFormatter(formatter) 
    # Apply the formatter to the console handler
    console_handler.setFormatter(formatter)  

    ### Add handlers to logger
    # Add the file handler to the logger, so it will write ERROR level messages to 'error.log'
    logger.addHandler(file_handler)  
    # Add the console handler to the logger, so it will write INFO level messages to the console
    logger.addHandler(console_handler)  

    # Now when you log messages, they are directed based on their severity:
    #logger.debug("This will print to console")
    #logger.info("This will also print to console")
    #logger.error("This will print to console and also save to error.log")

    return logger


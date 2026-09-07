import sys
from src.logger import logging

def error_message_details(error, error_detail: sys):
    _, _, exc_tb = error_detail.exc_info()
    error_message = f"Error occurred in python script name [{exc_tb.tb_frame.f_code.co_filename}] line number [{exc_tb.tb_lineno}] error message [{error}]"
    file_name = exc_tb.tb_frame.f_code.co_filename
    line_number = exc_tb.tb_lineno
    Error = str(error)

    return error_message



class CustomException(Exception):
    def __init__(self, error_message, error_detail:sys):
        super().__init__(error_message)
        self.error_message = error_message_details(error_message, error_detail = error_detail)


    def __str__(self):
        return self.error_message

if __name__ == "__main__":
    try:
        a = 1/0
    except Exception as e:
        logging.info("Divide by zero error")
        raise CustomException(e, sys)
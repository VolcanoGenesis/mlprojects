import sys
from src.logger import logging 

def error_message_detail(error, error_detail: sys):
    _, _, exe_tb = error_detail.exc_info()
    
    # Handle case where error is None or not a string
    error_str = str(error) if error is not None else "Unknown error"
    
    error_message = "Error occurred in python script name[{0}] line number[{1}] error message[{2}]".format(
        exe_tb.tb_frame.f_code.co_filename,
        exe_tb.tb_lineno,
        error_str
    )
    return error_message
        
class CustomerException(Exception):
    def __init__(self, error_message, error_detail: sys):
        super().__init__(error_message)
        self.error_message = error_message_detail(error_message, error_detail)

    def __str__(self):
        return self.error_message
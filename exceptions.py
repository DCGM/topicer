class CustomException(Exception):
    pass

def raise_exception():
    try:
         1 / 0
    except Exception as e:
        raise CustomException("This is a custom exception message.")

try:
    raise_exception()
except CustomException as ce:
    raise CustomException("Raising a new custom exception from the caught one.") 
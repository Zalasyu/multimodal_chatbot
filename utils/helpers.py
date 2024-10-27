def str_to_timestamp_milliseconds(time_str: str) -> float:
    """
    Convert a string in the format of "HH:MM:SS.sss" to a timestamp in milliseconds

    Args:
        time_str (str): The string to be converted

    Returns:
        float: The timestamp in milliseconds
    """
    return sum(60**x * int(y) for x, y in enumerate(reversed(time_str.split(":"))))

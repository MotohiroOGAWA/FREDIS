def charge_from_str(charge_str: str) -> int:
    """
    Convert a charge string to an integer.
    
    Args:
        charge_str (str): Charge string, e.g., "+", "-", "+2", "-3".
    
    Returns:
        int: Charge as an integer.
    """
    if charge_str == "":
        return 0
    elif charge_str == "+":
        return 1
    elif charge_str == "-":
        return -1
    elif charge_str.startswith("+"):
        return int(charge_str[1:])
    elif charge_str.startswith("-"):
        return int(charge_str)
    elif charge_str.endswith("+"):
        return int(charge_str[:-1])
    elif charge_str.endswith("-"):
        return -int(charge_str[:-1])
    else:
        raise ValueError(f"Invalid charge string: {charge_str}")
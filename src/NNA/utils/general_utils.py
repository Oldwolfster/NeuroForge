def ez_debug(**kwargs):
    """
    Print debug information for each provided variable.

    For every keyword argument passed in, this function prints:
    1) The variable name
    2) An equal sign
    3) The variable's value
    4) A tab character for separation

    Example:
        a = 1
        b = 2
        c = 3
        ez_debug(a=a, b=b, c=c)
        # Output: a=1    b=2    c=3
    """
    debug_output = ""
    for name, value in kwargs.items():
        debug_output += f"{name}={value}\t"
    print(debug_output)

def should_print_epoch(epoch: int, exp: int) -> bool:
    if exp>0:                   return False
    if epoch == 0:              return False            # preserve old "epoch != 0" behavior
    if epoch <= 50:             return True             # always print 1–10
    if epoch <= 100:            return epoch % 10 == 0  # 20, 30, 40, ... 100
    return epoch % 100 == 0                             # back to "normal" over 100



# Example usage:
if __name__ == "__main__":
    a = 1
    b = 2
    c = 3
    ez_debug(a=a, b=b, c=c)
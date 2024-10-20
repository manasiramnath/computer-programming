def bisec_search_rec(low, high, x, epsilon):
    """Assumes x and epsilon are numeric.
    Finds an approximation to the square root
    of a number x using bisection search.
    """

    # Start in the middle of the interval
    guess = (low + high) / 2

    if abs(guess**2 - x) < epsilon:
        return guess
    else:
        if guess**2 < x:
            return bisec_search_rec(guess, high, x, epsilon)
        else:
            return bisec_search_rec(low, guess, x, epsilon)


def bisec_search(x, epsilon):
    """Assumes x and epsilon are numeric.
    Finds an approximation to the square root
    of a number x using bisection search.
    """

    # Define interval for search
    low = 0
    high = max(1, x)

    # Start in the middle
    guess = (low + high) / 2

    # Narrow down search interval until guess close enough
    while abs(guess**2 - x) >= epsilon:
        if guess**2 < x:
            low = guess
        else:
            high = guess
        guess = (low + high) / 2

    return guess

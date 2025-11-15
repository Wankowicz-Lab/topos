def process_integers(numbers):
    """Remove odd integers and return the list with the average of remaining even integers."""
    evens = [n for n in numbers if n % 2 == 0]
    return evens + [sum(evens) / len(evens)] if evens else []

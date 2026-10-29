from XBDD import Ini_XBDD, apply, cofactor, create_var, is_true


def evaluate(dd, assignments):
    """Evaluate an XBDD using a mapping from variable levels to 0 or 1."""
    result = dd
    for level in sorted(assignments, reverse=True):
        result = cofactor(result, level, assignments[level])
    return int(is_true(result))


def main():
    Ini_XBDD()

    x0 = create_var(0)
    x1 = create_var(1)
    expression = apply(x0, x1, "AND")

    print("f(x0, x1) = x0 AND x1")
    print("x0 x1 | f")
    for x0_value in (0, 1):
        for x1_value in (0, 1):
            value = evaluate(expression, {0: x0_value, 1: x1_value})
            print(f" {x0_value}  {x1_value} | {value}")
    expression.show()

if __name__ == "__main__":
    main()

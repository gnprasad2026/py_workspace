from time import perf_counter

def timer(func):
    def wrapper(*args, **kwargs):
        start_time = perf_counter()
        result = func(*args, **kwargs)
        end_time = perf_counter()
        print(f"{func.__name__} time : {end_time - start_time:.6f} seconds")
        return result
    return wrapper

@timer
def sum_with_loop(n):
    total= 0
    for i in range(n):
        total+=i
    return total

@timer
def sum_with_formula(n):
    return n * (n-1) // 2

if __name__ == "__main__":
    n = 5000
    loop_result = sum_with_loop(n)
    formula_result = sum_with_formula(n)

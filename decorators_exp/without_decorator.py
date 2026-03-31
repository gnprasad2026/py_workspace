from time import perf_counter

def sum_with_loop(n):
    start_time1 = perf_counter()
    total= 0
    for i in range(n):
        total+=i
    end_time1 = perf_counter()

    print(f"sum_with_loop time : {end_time1 - start_time1:.6f} seconds")
    return total

def sum_with_formula(n):
    start_time =  perf_counter()
    result = n * (n-1) // 2
    end_time = perf_counter()

    print(f"sum with formula time : {end_time - start_time:.6f} seconds")
    return result

if __name__ == "__main__":
    n = 5000
    loop_result = sum_with_loop(n)
    formula_result = sum_with_formula(n)

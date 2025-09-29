import sys 

def generate_non_vectorized_design(vec_size):
    header = f"module linear(output wire [{vec_size - 1}:0] out, input wire [{vec_size - 1}:0] in);"
    assignments = ""
    for i in range(0, vec_size):
        assignments += f"\n{4 * ' '}assign out[{i}] = in[{i}];" 
    return f"{header}{assignments}\nendmodule"


def main(path):
    for i in range(1, 14):
        design = generate_non_vectorized_design(pow(2, i))
        with open(f"{path}/{i}-design.sv", "w", encoding="utf-8") as file: 
            file.write(design)




main(sys.argv[1])



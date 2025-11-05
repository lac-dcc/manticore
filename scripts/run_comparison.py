import os
import subprocess
import csv

PATH_ORIGINAL = "/chimera/chimera/database"
PATH_VETORIZADO = "/manticore/vectorized-files"

LISTA_DESIGNS_TXT = "vectorized-files.txt"
ARQUIVO_CSV_SAIDA = "results-time-memory.csv"

SCRIPT_TCL = "run_jasper.tcl"
SCRIPT_SHELL = "./measure.sh" 

def medir_desempenho(caminho_arquivo_design):
    """
    Executa o script de medição para um único arquivo de design e retorna o tempo e a memória.
    """
    if not os.path.exists(caminho_arquivo_design):
        print(f"AVISO: Arquivo não encontrado, pulando: {caminho_arquivo_design}")
        return "nao_encontrado", "nao_encontrado"


    # Cria uma cópia do ambiente atual e define a variável DESIGN_FILE
    env = os.environ.copy()
    env["DESIGN_FILE"] = caminho_arquivo_design

    try:
        # Chama o script shell, passando o script tcl como argumento
        # Captura a saída (stdout) e erros (stderr)
        # 'env=env' passa a variável de ambiente que definimos
        processo = subprocess.run(
            [SCRIPT_SHELL, SCRIPT_TCL],
            capture_output=True,
            text=True,
            check=True,
            env=env
        )

        # A saída do script shell são duas linhas:
        # 1. tempo total (em microssegundos)
        # 2. memória (em KB)
        output_lines = processo.stdout.strip().split('\n')
        
        # Converte tempo de microssegundos para segundos
        tempo_us = float(output_lines[0])
        tempo_s = tempo_us / 1_000_000
        
        memoria_kb = int(output_lines[1])
        
        return tempo_s, memoria_kb

    except subprocess.CalledProcessError as e:
        print(f"ERRO ao processar {caminho_arquivo_design}:")
        print(e.stderr)
        return "erro", "erro"
    except (IndexError, ValueError) as e:
        print(f"ERRO ao parsear a saída para {caminho_arquivo_design}: {e}")
        return "erro_parse", "erro_parse"


def main():

    with open(ARQUIVO_CSV_SAIDA, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([
            "nome_do_arquivo", 
            "tempo_original_s", 
            "tempo_vetorizado_s", 
            "memoria_original_kb", 
            "memoria_vetorizada_kb"
        ])

        with open(LISTA_DESIGNS_TXT, 'r') as f:
            for nome_base_arquivo in f:
                nome_base_arquivo = nome_base_arquivo.strip()
                if not nome_base_arquivo:
                    continue

                caminho_original = os.path.join(PATH_ORIGINAL, nome_base_arquivo)
                caminho_vetorizado = os.path.join(PATH_VETORIZADO, nome_base_arquivo)

                tempo_orig, mem_orig = medir_desempenho(caminho_original)
                tempo_vet, mem_vet = medir_desempenho(caminho_vetorizado)
                
                writer.writerow([nome_base_arquivo, tempo_orig, tempo_vet, mem_orig, mem_vet])
    
    print(f"\nResultados salvos em '{ARQUIVO_CSV_SAIDA}'")

if __name__ == "__main__":
    main()
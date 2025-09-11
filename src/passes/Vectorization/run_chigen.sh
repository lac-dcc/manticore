#!/bin/bash

# Caminhos importantes
DB_DIR=~/chimera/chimera/database
GENUS_OUT=~/chimera/chimera/processed
VEC_OUT=~/chimera/chimera/vectorized
RUN_VEC=~/manticore/src/passes/Vectorization/run_vec.sh

mkdir -p $GENUS_OUT
mkdir -p $VEC_OUT

process_file() {
    FILE="$1"
    BASENAME=$(basename "$FILE" .sv)

    echo "🔹 [$$] Processando $BASENAME.sv"

    # Arquivo TCL único para este arquivo
    GENUS_CMD=$(mktemp)
    cat > "$GENUS_CMD" <<EOF
set_db library tutorial.lib
read_hdl -language sv $FILE
elaborate $BASENAME
ungroup -all -flatten
syn_generic
syn_map
write_netlist > $GENUS_OUT/${BASENAME}_flattened.v
exit
EOF

    # Rodar Genus
    genus -batch -files "$GENUS_CMD" > "$GENUS_OUT/${BASENAME}.log" 2>&1
    rm -f "$GENUS_CMD"

    # Se o arquivo flatten existir, roda vetorização
    if [ -f "$GENUS_OUT/${BASENAME}_flattened.v" ]; then
        echo "🔹 [$$] Vetorizando $BASENAME"
        "$RUN_VEC" "$GENUS_OUT/${BASENAME}_flattened.v" "$VEC_OUT/${BASENAME}_final.v" \
            > "$VEC_OUT/${BASENAME}.log" 2>&1
    else
        echo "⚠️ [$BASENAME] Genus falhou, pulando vetorização."
    fi
}

export -f process_file
export GENUS_OUT VEC_OUT RUN_VEC

# Rodar com paralelismo (ajuste -P para usar mais/menos núcleos)
find "$DB_DIR" -name "*.sv" | parallel -j 8 process_file {}

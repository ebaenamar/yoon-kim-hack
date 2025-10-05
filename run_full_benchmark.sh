#!/bin/bash
#
# Script robusto para lanzar el benchmark completo.
# Activa el entorno de Conda y luego ejecuta el script de Python.

echo "Lanzando el benchmark completo..."

# Asegúrate de que Conda está inicializado
source /home/nuwins-server1/anaconda3/etc/profile.d/conda.sh

# Activa el entorno
conda activate arch_bench

# Lanza el script de Python
# Ejecutará todas las arquitecturas y tareas por defecto, con 10 rondas.
python run_benchmark.py --num-runs 10

echo "El benchmark ha finalizado."

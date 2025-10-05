#!/usr/bin/env python
"""Reporte completo de instalación"""

import sys

def print_section(title):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print('='*70)

print_section("VERIFICACIÓN COMPLETA DE INSTALACIÓN")

# 1. Sistema y GPU
print_section("1. SISTEMA Y GPU")
try:
    import torch
    print(f"✅ PyTorch versión: {torch.__version__}")
    print(f"✅ CUDA disponible: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
        mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"✅ Memoria GPU: {mem:.1f} GB")
except Exception as e:
    print(f"❌ Error: {e}")

# 2. Dependencias básicas
print_section("2. DEPENDENCIAS BÁSICAS")
deps = {
    'torch': 'PyTorch',
    'einops': 'Einops',
    'transformers': 'Transformers',
    'datasets': 'Datasets',
    'numpy': 'NumPy',
    'pandas': 'Pandas',
    'matplotlib': 'Matplotlib',
    'seaborn': 'Seaborn',
    'tqdm': 'TQDM'
}

all_deps_ok = True
for module, name in deps.items():
    try:
        __import__(module)
        print(f"✅ {name}")
    except ImportError:
        print(f"❌ {name} - falta instalar")
        all_deps_ok = False

# 3. Tareas (Tasks)
print_section("3. TAREAS DE BENCHMARK")
try:
    from tasks import MQARDataset, MQARConfig
    from tasks import FlipFlopDataset, FlipFlopConfig
    from tasks import KnapsackTask, TropicalTaskConfig
    
    # Test MQAR
    config = MQARConfig(vocab_size=20, num_kv_pairs=5, num_queries=3)
    ds = MQARDataset(config)
    inp, tgt, mask = ds.generate_sample()
    print(f"✅ MQAR - Secuencia generada: {len(inp)} tokens")
    
    # Test Flip-Flop
    config2 = FlipFlopConfig(num_symbols=5, seq_length=20)
    ds2 = FlipFlopDataset(config2)
    inp2, tgt2, mask2 = ds2.generate_sample()
    print(f"✅ Flip-Flop - Secuencia generada: {len(inp2)} tokens")
    
    # Test Knapsack
    config3 = TropicalTaskConfig()
    task = KnapsackTask(config3)
    inp3, tgt3, info = task.generate_sample()
    print(f"✅ Knapsack - Problema generado: capacidad={info['capacity']}")
    
except Exception as e:
    print(f"❌ Error en tareas: {e}")

# 4. Modelos
print_section("4. ARQUITECTURAS DE MODELOS")
try:
    from models.model_factory import list_available_architectures, create_model
    from models.base_model import ModelConfig
    
    archs = list_available_architectures()
    print(f"Total de arquitecturas disponibles: {len(archs)}\n")
    
    working_archs = []
    fla_archs = []
    
    for arch in archs:
        try:
            config = ModelConfig(vocab_size=50, d_model=128, n_layers=2)
            model = create_model(arch, config)
            params = model.get_num_params()
            print(f"✅ {arch.upper():20s} - {params:>10,} parámetros")
            working_archs.append(arch)
        except Exception as e:
            if 'fla' in str(e).lower():
                print(f"⚠️  {arch.upper():20s} - Requiere FlashLinearAttention")
                fla_archs.append(arch)
            else:
                print(f"❌ {arch.upper():20s} - Error: {e}")
    
    print(f"\n📊 Arquitecturas funcionando: {len(working_archs)}")
    print(f"📦 Arquitecturas que requieren FLA: {len(fla_archs)}")
    
except Exception as e:
    print(f"❌ Error en modelos: {e}")
    import traceback
    traceback.print_exc()

# 5. FlashLinearAttention
print_section("5. FLASHLINEARATTENTION (Opcional)")
fla_ok = True
fla_modules = ['DeltaNet', 'GatedDeltaNet', 'RWKV7Attention', 'MultiScaleRetention', 'GatedLinearAttention']
for module in fla_modules:
    try:
        exec(f"from fla.layers import {module}")
        print(f"✅ {module}")
    except ImportError:
        print(f"⚠️  {module} - No disponible")
        fla_ok = False

if not fla_ok:
    print("\nℹ️  FlashLinearAttention no está instalado (opcional)")
    print("   Para instalar: bash install_fla.sh")

# 6. Sistema de entrenamiento
print_section("6. SISTEMA DE ENTRENAMIENTO")
try:
    from trainer import train_model, SequenceDataset
    print("✅ Trainer")
    print("✅ SequenceDataset")
except Exception as e:
    print(f"❌ Error: {e}")

# 7. Runner de benchmarks
print_section("7. RUNNER DE BENCHMARKS")
try:
    import os
    files_to_check = [
        'run_benchmark.py',
        'quick_start.py',
        'trainer.py'
    ]
    for f in files_to_check:
        if os.path.exists(f):
            print(f"✅ {f}")
        else:
            print(f"❌ {f} - No encontrado")
except Exception as e:
    print(f"❌ Error: {e}")

# RESUMEN FINAL
print_section("📋 RESUMEN FINAL")

if all_deps_ok:
    print("✅ Todas las dependencias básicas instaladas")
else:
    print("⚠️  Faltan algunas dependencias básicas")

print(f"\n🏗️  Arquitecturas listas para usar: {len(working_archs)}")
print("   " + ", ".join([a.upper() for a in working_archs]))

if fla_archs:
    print(f"\n📦 Arquitecturas que requieren FLA: {len(fla_archs)}")
    print("   " + ", ".join([a.upper() for a in fla_archs]))

print("\n" + "="*70)
if len(working_archs) >= 3:
    print("✅ ¡SISTEMA LISTO PARA EJECUTAR BENCHMARKS!")
    print("\nPuedes ejecutar:")
    print("  python quick_start.py")
    print("  python run_benchmark.py --quick")
else:
    print("⚠️  Faltan componentes críticos")

if not fla_ok:
    print("\nPara instalar FlashLinearAttention:")
    print("  bash install_fla.sh")
    
print("="*70 + "\n")

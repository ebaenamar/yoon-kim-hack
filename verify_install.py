"""Verify installation of all components"""

import sys

print("="*60)
print("Installation Verification")
print("="*60)

# Check PyTorch
try:
    import torch
    print(f"✓ PyTorch {torch.__version__}")
    print(f"  CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
except ImportError as e:
    print(f"✗ PyTorch not installed: {e}")
    sys.exit(1)

# Check basic dependencies
deps = ['einops', 'transformers', 'datasets', 'tqdm', 'pandas', 'matplotlib', 'seaborn', 'numpy']
for dep in deps:
    try:
        __import__(dep)
        print(f"✓ {dep}")
    except ImportError:
        print(f"✗ {dep} - run: pip install {dep}")

# Check FlashLinearAttention
print("\n" + "="*60)
print("FlashLinearAttention Components")
print("="*60)

fla_available = True
try:
    from fla.layers import DeltaNet
    print("✓ DeltaNet")
except ImportError as e:
    print(f"✗ DeltaNet: {e}")
    fla_available = False

try:
    from fla.layers import GatedDeltaNet
    print("✓ Gated DeltaNet")
except ImportError as e:
    print(f"✗ Gated DeltaNet: {e}")
    fla_available = False

try:
    from fla.layers import RWKV7Attention
    print("✓ RWKV-7")
except ImportError as e:
    print(f"✗ RWKV-7: {e}")
    fla_available = False

try:
    from fla.layers import MultiScaleRetention
    print("✓ Retention")
except ImportError as e:
    print(f"✗ Retention: {e}")
    fla_available = False

try:
    from fla.layers import GatedLinearAttention
    print("✓ GLA")
except ImportError as e:
    print(f"✗ GLA: {e}")
    fla_available = False

# Check our modules
print("\n" + "="*60)
print("Benchmark Components")
print("="*60)

try:
    from tasks import MQARDataset, FlipFlopDataset, KnapsackTask
    print("✓ All tasks")
except ImportError as e:
    print(f"✗ Tasks: {e}")

try:
    from models.model_factory import list_available_architectures
    archs = list_available_architectures()
    print(f"✓ Model factory - {len(archs)} architectures available")
    print(f"  Architectures: {', '.join(archs)}")
except ImportError as e:
    print(f"✗ Model factory: {e}")

# Summary
print("\n" + "="*60)
print("Summary")
print("="*60)

if fla_available:
    print("✅ Full installation complete!")
    print("   All 9 architectures available")
    print("\n   Ready to run:")
    print("   python run_benchmark.py --all")
else:
    print("⚠️  Basic installation complete (4 architectures)")
    print("   FlashLinearAttention not installed")
    print("\n   You can run:")
    print("   python run_benchmark.py --architectures lstm,gru,rope --tasks mqar,flip_flop")
    print("\n   To install FLA:")
    print("   pip install -U git+https://github.com/fla-org/flash-linear-attention")

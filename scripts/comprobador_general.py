"""
General Health Check — Runs all phase verifiers and reports overall status.
Exit code 0 = all phases passed, 1 = at least one failed.
"""
import sys
import subprocess
from pathlib import Path


PHASES = [
    ("Fase 0: Setup", "scripts/fase0_setup_verificar.py"),
    ("Fase 1: Datos", "scripts/fase1_datos_verificar.py"),
    ("Fase 2: Entrenamiento", "scripts/fase2_entrenamiento_verificar.py"),
    ("Fase 3: Evaluacion", "scripts/fase3_evaluacion_verificar.py"),
    ("Fase 4: Inferencia", "scripts/fase4_inferencia_verificar.py"),
    ("Fase 5: Exportacion", "scripts/fase5_exportacion_verificar.py"),
    ("Fase 6: Demo", "scripts/fase6_demo_verificar.py"),
]


def run_all_checks():
    print("=" * 60)
    print("  COMPROBADOR GENERAL — YOLOv8 Pipeline Health Check")
    print("=" * 60)
    print()

    all_passed = True
    results_summary = []

    for name, script in PHASES:
        script_path = Path(script)
        if not script_path.exists():
            print(f"  ⚠️  {name}: script not found ({script})")
            results_summary.append((name, "SKIP"))
            continue

        try:
            result = subprocess.run(
                [sys.executable, str(script_path)],
                capture_output=True,
                text=True,
                timeout=60,
            )
            if result.returncode == 0:
                print(f"  ✅ {name}: PASS")
                results_summary.append((name, "PASS"))
            else:
                print(f"  ❌ {name}: FAIL")
                all_passed = False
                results_summary.append((name, "FAIL"))
        except subprocess.TimeoutExpired:
            print(f"  ⏱️  {name}: TIMEOUT")
            all_passed = False
            results_summary.append((name, "TIMEOUT"))
        except Exception as e:
            print(f"  ❌ {name}: ERROR ({e})")
            all_passed = False
            results_summary.append((name, "ERROR"))

    print()
    print("=" * 60)
    if all_passed:
        print("  ✅ RESULTADO: COMPLETO — Todas las fases aprobadas")
    else:
        failed = [name for name, status in results_summary if status != "PASS"]
        print(f"  ❌ RESULTADO: INCOMPLETO — Fases fallidas: {', '.join(failed)}")
    print("=" * 60)

    return all_passed


if __name__ == "__main__":
    ok = run_all_checks()
    sys.exit(0 if ok else 1)

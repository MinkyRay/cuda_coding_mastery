import os
import subprocess
import re
import matplotlib.pyplot as plt
import sys
import math

SOURCE_FILE = "../src/04_gemm_double_buffering.cu"
OUTPUT_EXE = "bin_sensitivity"
if sys.platform == "win32":
    OUTPUT_EXE += ".exe"

BLOCK_SIZES = [4, 8, 16]


BM, BN = 64, 64
M, N, K = 4096, 4096, 4096

NVCC_FLAGS = "-arch=sm_89 -O3"

def run_sensitivity_test():
    results = []
    valid_labels = []
    
    print(f"Starting Sensitivity Analysis...")
    print(f"Fixed Macro-Tile: BM={BM}, BN={BN}")
    print(f"Matrix Size: {M}x{N}x{K}\n")

    for size in BLOCK_SIZES:
        tm, tn = size, size
        
        num_threads = (BM * BN) / (tm * tn)
        
        print(f"--- Checking Config: TM={tm}, TN={tn} ---")
        

        if num_threads > 1024:
            print(f"[Skip] Requires {int(num_threads)} threads (Max 1024).")
            print(f"       Hint: Reduce BM/BN to run this size.")
            results.append(0)
            continue
            
        
        if (BM % tm != 0) or (BN % tn != 0):
            print(f"[Skip] BM({BM}) not divisible by TM({tm}). Kernel logic will fail.")
            results.append(0)
            continue
            
        valid_labels.append(f"{tm}x{tn}")

        compile_cmd = f"nvcc {SOURCE_FILE} -o {OUTPUT_EXE} {NVCC_FLAGS} -D BM={BM} -D BN={BN} -D TM={tm} -D TN={tn}"
        
        print(f"Compiling...")
        ret = subprocess.call(compile_cmd, shell=True)
        if ret != 0:
            print(f"[Error] Compile failed for {tm}x{tn}")
            results.append(0)
            continue

        run_cmd = f"{OUTPUT_EXE} {M} {N} {K}"
        try:
            output = subprocess.check_output(run_cmd, shell=True).decode("utf-8", errors="ignore")
            match = re.search(r"Performance:\s+([\d\.]+)\s+GFLOPS", output)
            if match:
                gflops = float(match.group(1))
                print(f"-> Result: {gflops:.2f} GFLOPS")
                results.append(gflops)
            else:
                print("-> Parse Error (Could not find GFLOPS in output)")
                results.append(0)
        except Exception as e:
            print(f"-> Runtime Error or Kernel Launch Failure (Reg Spill?): {e}")
            results.append(0)
            
    return valid_labels, [r for r in results if r > 0]


def plot_sensitivity(labels, scores):
    if not scores:
        print("No valid results to plot.")
        return

    plt.figure(figsize=(10, 6))
    
    plt.plot(labels, scores, marker='o', linewidth=2, color='#d62728', label='Performance Trend')
    

    max_gflops = max(scores)
    max_idx = scores.index(max_gflops)
    plt.annotate(f'Sweet Spot: {max_gflops:.0f} GFLOPS', 
                 xy=(max_idx, max_gflops), 
                 xytext=(max_idx, max_gflops + 500),
                 arrowprops=dict(facecolor='black', shrink=0.05),
                 fontsize=12, fontweight='bold')

    plt.title(f"Register Block Sensitivity (BM={BM}, BN={BN})", fontsize=16)
    plt.xlabel("Register Block Size (TM x TN)", fontsize=14)
    plt.ylabel("Performance (GFLOPS)", fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    output_file = "register_sensitivity.png"
    plt.savefig(output_file)
    print(f"\n[Success] Graph saved to {output_file}")

if __name__ == "__main__":
    labels, data = run_sensitivity_test()
    plot_sensitivity(labels, data)
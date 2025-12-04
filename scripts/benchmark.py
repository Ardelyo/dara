"""
DARA Benchmark Script
Runs performance tests and generates research-quality statistics.
"""

import time
import json
import sys
import os
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import Optional
import statistics

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


@dataclass
class BenchmarkResult:
    """Single benchmark test result."""
    mode: str
    image_path: str
    inference_time_ms: float
    confidence: float
    result_length: int
    success: bool
    error: Optional[str] = None


@dataclass
class BenchmarkSummary:
    """Summary statistics for benchmarks."""
    total_tests: int
    successful_tests: int
    failed_tests: int
    avg_inference_time_ms: float
    min_inference_time_ms: float
    max_inference_time_ms: float
    std_inference_time_ms: float
    avg_confidence: float
    mode_breakdown: dict
    timestamp: str
    device: str
    model_id: str


def run_benchmark(
    image_paths: list,
    modes: list = None,
    iterations: int = 1,
    warmup: int = 1
) -> BenchmarkSummary:
    """
    Run comprehensive benchmark tests.
    
    Args:
        image_paths: List of image paths to test
        modes: List of modes to test (default: all)
        iterations: Number of iterations per image/mode
        warmup: Number of warmup runs
        
    Returns:
        BenchmarkSummary with statistics
    """
    from dara import DARA, Config
    
    print("=" * 60)
    print("DARA BENCHMARK TEST")
    print("=" * 60)
    
    # Initialize model
    print("\n📦 Loading DARA model...")
    start_load = time.time()
    dara = DARA(enable_cache=False)  # Disable cache for accurate timing
    load_time = time.time() - start_load
    print(f"   Model loaded in {load_time:.2f}s")
    print(f"   Device: {dara.device}")
    print(f"   Model: {dara.model_id}")
    
    # Get modes
    if modes is None:
        modes = dara.get_available_modes()
    
    print(f"\n🎯 Testing modes: {modes}")
    print(f"📷 Images: {len(image_paths)}")
    print(f"🔄 Iterations: {iterations}")
    
    # Warmup
    if warmup > 0 and image_paths:
        print(f"\n🔥 Warmup ({warmup} runs)...")
        for _ in range(warmup):
            try:
                dara.detect(image_paths[0], mode=modes[0], generate_audio=False)
            except:
                pass
    
    # Run benchmarks
    results: list[BenchmarkResult] = []
    
    print(f"\n🏃 Running benchmarks...")
    
    for img_path in image_paths:
        for mode in modes:
            for i in range(iterations):
                try:
                    start = time.perf_counter()
                    result = dara.detect(
                        img_path, 
                        mode=mode, 
                        generate_audio=False
                    )
                    elapsed = (time.perf_counter() - start) * 1000
                    
                    results.append(BenchmarkResult(
                        mode=mode,
                        image_path=str(img_path),
                        inference_time_ms=elapsed,
                        confidence=result.get("confidence", 0),
                        result_length=len(result.get("result", "")),
                        success=True
                    ))
                    
                    print(f"   ✓ {mode}: {elapsed:.1f}ms (conf: {result.get('confidence', 0):.2f})")
                    
                except Exception as e:
                    results.append(BenchmarkResult(
                        mode=mode,
                        image_path=str(img_path),
                        inference_time_ms=0,
                        confidence=0,
                        result_length=0,
                        success=False,
                        error=str(e)
                    ))
                    print(f"   ✗ {mode}: ERROR - {e}")
    
    # Calculate statistics
    successful = [r for r in results if r.success]
    times = [r.inference_time_ms for r in successful]
    confidences = [r.confidence for r in successful]
    
    # Mode breakdown
    mode_stats = {}
    for mode in modes:
        mode_results = [r for r in successful if r.mode == mode]
        if mode_results:
            mode_times = [r.inference_time_ms for r in mode_results]
            mode_confs = [r.confidence for r in mode_results]
            mode_stats[mode] = {
                "count": len(mode_results),
                "avg_time_ms": statistics.mean(mode_times),
                "avg_confidence": statistics.mean(mode_confs)
            }
    
    summary = BenchmarkSummary(
        total_tests=len(results),
        successful_tests=len(successful),
        failed_tests=len(results) - len(successful),
        avg_inference_time_ms=statistics.mean(times) if times else 0,
        min_inference_time_ms=min(times) if times else 0,
        max_inference_time_ms=max(times) if times else 0,
        std_inference_time_ms=statistics.stdev(times) if len(times) > 1 else 0,
        avg_confidence=statistics.mean(confidences) if confidences else 0,
        mode_breakdown=mode_stats,
        timestamp=datetime.now().isoformat(),
        device=dara.device,
        model_id=dara.model_id
    )
    
    # Print summary
    print("\n" + "=" * 60)
    print("📊 BENCHMARK RESULTS")
    print("=" * 60)
    print(f"Total Tests:     {summary.total_tests}")
    print(f"Successful:      {summary.successful_tests}")
    print(f"Failed:          {summary.failed_tests}")
    print(f"\n⏱️  Inference Time:")
    print(f"   Average:      {summary.avg_inference_time_ms:.1f} ms")
    print(f"   Min:          {summary.min_inference_time_ms:.1f} ms")
    print(f"   Max:          {summary.max_inference_time_ms:.1f} ms")
    print(f"   Std Dev:      {summary.std_inference_time_ms:.1f} ms")
    print(f"\n🎯 Confidence:   {summary.avg_confidence:.3f}")
    print(f"\n📈 Mode Breakdown:")
    for mode, stats in mode_stats.items():
        print(f"   {mode}: {stats['avg_time_ms']:.1f}ms, conf={stats['avg_confidence']:.3f}")
    
    return summary


def save_results(summary: BenchmarkSummary, output_path: str):
    """Save benchmark results to JSON."""
    with open(output_path, "w") as f:
        json.dump(asdict(summary), f, indent=2)
    print(f"\n💾 Results saved to: {output_path}")


if __name__ == "__main__":
    # Find test images
    project_root = Path(__file__).parent.parent
    sample_dir = project_root / "sampleimages"
    
    # Look for any images
    image_paths = []
    for ext in ["*.jpg", "*.jpeg", "*.png"]:
        image_paths.extend(sample_dir.glob(ext))
    
    if not image_paths:
        print("⚠️  No sample images found in sampleimages/")
        print("   Creating a test with a placeholder...")
        # Create minimal test
        image_paths = []
    
    if image_paths:
        summary = run_benchmark(
            image_paths=list(image_paths)[:5],  # Limit to 5 images
            modes=["scene", "currency", "text"],
            iterations=1,
            warmup=1
        )
        
        # Save results
        output_path = project_root / "docs" / "benchmark_results.json"
        save_results(summary, str(output_path))
    else:
        print("\n📝 To run benchmarks, add images to sampleimages/ folder")

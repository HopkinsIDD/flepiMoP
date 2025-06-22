#/flepiMoP/flepimop/gempyor_pkg/src/gempyor/benchmark_utils.py
def save_benchmark_result(
    impl,
    ncompartments,
    nspatial_nodes,
    ndays,
    elapsed,
    peak_memory,
    result_shape,
    cap_time=None,
    fn_hash=None
):
    peak_MB = peak_memory / 1024**2
    capped = elapsed > cap_time if cap_time else False
    status = "ok"
    if capped:
        status = "fail_time"
    if peak_MB > 500.0:
        status = "fail_memory"
    if elapsed > cap_time and result_shape == "n/a":
        status = "exception"

    record = {
        "timestamp": datetime.now().isoformat(),
        "impl": impl,
        "ncompartments": int(ncompartments),
        "nspatial_nodes": int(nspatial_nodes),
        "ndays": int(ndays),
        "wall_time": round(min(elapsed, cap_time) if cap_time else elapsed, 4),
        "peak_memory_MB": round(peak_MB, 2),
        "result_shape": str(result_shape),
        "status": status,
        "fn_hash": fn_hash,
        **ENV_INFO
    }

    with open(RESULTS_FILE, "a") as f:
        f.write(json.dumps(record) + "\n")
